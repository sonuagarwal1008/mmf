# Copyright (c) Facebook, Inc. and its affiliates.

import os
import unittest

import torch
from mmf.common.registry import registry
from mmf.trainers.mmf_trainer import MMFTrainer
from mmf.utils.build import build_optimizer
from omegaconf import OmegaConf
from tests.test_utils import SimpleModel, skip_if_no_cuda
from tests.trainers.test_training_loop import TrainerTrainingLoopMock


try:
    from fairscale.optim.oss import OSS
    from fairscale.nn.data_parallel import ShardedDataParallel
    from fairscale.optim.grad_scaler import ShardedGradScaler

    FAIRSCALE_AVAILABLE = True
except ImportError:
    FAIRSCALE_AVAILABLE = False


class MMFTrainerMock(TrainerTrainingLoopMock, MMFTrainer):
    def __init__(
        self,
        config,
        num_train_data,
        max_updates,
        max_epochs,
        device="cuda",
        fp16_model=False,
    ):
        super().__init__(num_train_data, max_updates, max_epochs, fp16=True)
        self.device = torch.device(device)
        self.config = config
        self.model = SimpleModel({"in_dim": 1})
        self.model.build()
        self.model = self.model.cuda()
        self.optimizer = build_optimizer(self.model, self.config)
        self.distributed = True
        self.local_rank = 0
        self.parallelize_model()
        self.load_fp16_scaler()


class TestShardedDDP(unittest.TestCase):
    def setUp(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        torch.distributed.init_process_group(
            backend=torch.distributed.Backend.NCCL, rank=0, world_size=1
        )
        self.config_oss = OmegaConf.create(
            {
                "optimizer": {
                    "type": "adam_w",
                    "enable_state_sharding": True,
                    "params": {"lr": 5e-5},
                },
                "training": {"find_unused_parameters": False},
            }
        )
        self.config_no_oss = OmegaConf.create(
            {
                "optimizer": {
                    "type": "adam_w",
                    "enable_state_sharding": False,
                    "params": {"lr": 5e-5},
                },
                "training": {"find_unused_parameters": False},
            }
        )
        self.trainer = None

    def tearDown(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        del self.trainer
        registry.unregister("distributed")

    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    def test_no_sharding(self):
        self.trainer = MMFTrainerMock(self.config_no_oss, 100, 2, 0.04)

        self.assertFalse(isinstance(self.trainer.optimizer, OSS))

        self.assertFalse(isinstance(self.trainer.model, ShardedDataParallel))
        self.assertTrue(
            isinstance(self.trainer.model, torch.nn.parallel.DistributedDataParallel)
        )

        self.assertFalse(isinstance(self.trainer.scaler, ShardedGradScaler))
        self.assertTrue(isinstance(self.trainer.scaler, torch.cuda.amp.GradScaler))

        self.assertEqual(self.trainer.current_iteration, 0)
        self.trainer.training_loop()
        self.assertEqual(self.trainer.current_iteration, 4)

    @skip_if_no_cuda
    @unittest.skipUnless(FAIRSCALE_AVAILABLE, "Tests for fairscale")
    def test_sharding(self):
        self.trainer = MMFTrainerMock(self.config_oss, 100, 2, 0.04)

        self.assertTrue(isinstance(self.trainer.optimizer, OSS))

        self.assertTrue(isinstance(self.trainer.model, ShardedDataParallel))

        self.assertTrue(isinstance(self.trainer.scaler, ShardedGradScaler))

        self.assertEqual(self.trainer.current_iteration, 0)
        self.trainer.training_loop()
        self.assertEqual(self.trainer.current_iteration, 4)
