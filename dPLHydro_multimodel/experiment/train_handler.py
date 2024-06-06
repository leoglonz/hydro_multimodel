import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from conf.config import Config
from experiment.experiment_tracker import Tracker
from injector import inject
from models.neural_networks import NeuralNetwork
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dPLHydro_multimodel.data.general_dataset import GeneralDataset

# from models.neural_networks.criterion.mean_range_bound_loss import MeanRangeBoundLoss
# from models.neural_networks.criterion.rmse import RMSELoss

log = logging.getLogger(__name__)


class TrainHandler:
    @inject
    def __init__(
        self,
        dataset: GeneralDataset,
        neural_networks: Dict[str, NeuralNetwork],
        physics_models: Dict[str, PhysicsModel],
    ):
        self.dataset = dataset
        self.neural_networks = neural_networks
        self.physics_models = physics_models

    def run(self, cfg: Config, experiment_tracker: Tracker) -> None:
        world_size = cfg.world_size
        rank = cfg.local_rank
        device = cfg.device[rank]

        log.info(f"Training model: {cfg.name}")
        warmup = cfg.params.warmup
        mlp = self.neural_networks["mlp"]
        streamflow = self.physics_models["streamflow"]
        dmc = self.physics_models["dMC"]

        criterion = RMSELoss()
        range_bound_loss = MeanRangeBoundLoss(cfg)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=cfg.train.learning_rate)

        if cfg.train.checkpoint is None:
            log.info("Initializing new model")
            mlp.initialize_weights()
            start_epoch = 1
        else:
            file_path = Path(cfg.train.checkpoint)
            log.info(f"Loading from checkpoint: {file_path.stem}")
            state = torch.load(file_path)
            mlp.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            torch.set_rng_state(state["rng_state"])
            start_epoch = state["epoch"]
            if torch.cuda.is_available() and "cuda_rng_state" in state:
                torch.cuda.set_rng_state(state["cuda_rng_state"])

        # Set devices and data parallel
        mlp.layers = mlp.layers.to(device)
        mlp = DDP(mlp, device_ids=[device])

        sampler = DistributedSampler(
            dataset=self.dataset, num_replicas=world_size, rank=rank
        )

        dataloader = DataLoader(
            dataset=self.dataset,
            # shuffle=cfg.train.shuffle,
            batch_size=cfg.train.batch_size,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
            sampler=sampler,
        )
        loss_idx_value = 0
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            dmc.epoch = epoch
            total_loss = 0.0
            sampler.set_epoch(epoch)
            for i, hydrofabric in enumerate(dataloader):
                dist.barrier()
                dmc.mini_batch = i

                streamflow_predictions = streamflow(cfg=cfg, hydrofabric=hydrofabric)
                nn_output = mlp(inputs=hydrofabric.normalized_attributes.to(device))
                q_prime = streamflow_predictions["streamflow"] @ hydrofabric.mapping.tm
                dmc_output = dmc(
                    hydrofabric=hydrofabric, parameters=nn_output, streamflow=q_prime
                )

                nan_mask = hydrofabric.observations.isnull().any(dim="time")
                np_nan_mask = nan_mask.streamflow.values

                filtered_ds = hydrofabric.observations.where(~nan_mask, drop=True)
                filtered_observations = torch.tensor(
                    filtered_ds.streamflow.values, device=device
                )

                filtered_predictions = dmc_output["runoff"][~np_nan_mask]
                gage_indices = np.array(
                    object=hydrofabric.network.gage_information["gage_dict_idx"]
                )
                gage_ids = np.array(self.dataset.gage_dict["STAID"])
                filtered_gage_indices = np.array(gage_indices)[~np_nan_mask]
                filtered_gage_ids = gage_ids[filtered_gage_indices]

                l1 = 0
                for j in range(filtered_observations.shape[0]):
                    l1 = l1 + criterion(
                        prediction=filtered_predictions[j][warmup:],
                        target=filtered_observations[j][warmup:],
                        gage=filtered_gage_ids[j],
                    )
                l1 = l1 / filtered_observations.shape[0]
                l2 = range_bound_loss(
                    [dmc.n, dmc.q_spatial],
                )
                loss = l1 + l2

                dist.barrier()
                if rank == 0:
                    log.info("Running gradient-averaged backpropagation")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if rank == 0:
                    experiment_tracker.set_metrics(
                        pred=filtered_predictions.detach().cpu().numpy(),
                        target=filtered_observations.detach().cpu().numpy(),
                    )
                    experiment_tracker.plot_all_time_series(
                        filtered_predictions.detach().cpu().numpy()[..., warmup:],
                        filtered_observations.cpu().numpy()[..., warmup:],
                        self.dataset.gage_dict,
                        self.dataset.dates.batch_hourly_time_range[..., warmup:],
                        filtered_gage_indices,
                        cfg.mode,
                        epoch,
                    )
                    total_loss = total_loss + loss.item()
                    pred_nse = experiment_tracker.metrics.nse
                    pred_nse_filtered = pred_nse[
                        ~np.isinf(pred_nse) & ~np.isnan(pred_nse)
                    ]
                    median_nse = torch.tensor(pred_nse_filtered).median()
                    experiment_tracker.writer.add_scalar(
                        "Median NSE", median_nse, loss_idx_value
                    )
                    experiment_tracker.writer.add_scalar(
                        "Median Manning's n",
                        dmc.n.detach().mean().item(),
                        loss_idx_value,
                    )
                    experiment_tracker.writer.add_scalar(
                        "Median spatial q",
                        dmc.q_spatial.detach().mean().item(),
                        loss_idx_value,
                    )
                    areas = hydrofabric.attributes[
                        :, cfg.params.attribute_indices.area
                    ].numpy()
                    experiment_tracker.writer.add_figure(
                        "Manning's n distribution",
                        experiment_tracker.plot_parameter_distribution(
                            x_data=areas,
                            x_label=r"Drainage Area $(km^2)$",
                            y_data=dmc.n.detach().cpu().numpy(),
                            y_label=r"Manning's n $(m^{1/3}/s)$",
                        ),
                        loss_idx_value,
                    )
                    random_index = np.random.randint(
                        low=0, high=filtered_gage_indices.shape[0], size=(1,)
                    )[0]
                    random_gage = filtered_gage_indices[random_index]
                    experiment_tracker.writer.add_figure(
                        "Trained Routing Models",
                        experiment_tracker.plot_time_series(
                            filtered_predictions[random_index].detach().cpu().numpy(),
                            filtered_observations[random_index].cpu().numpy(),
                            self.dataset.dates.batch_hourly_time_range,
                            self.dataset.gage_dict["STAID"][random_gage],
                            self.dataset.gage_dict["STANAME"][random_gage],
                            mode=cfg.mode,
                            metrics={"nse": pred_nse[random_index]},
                        ),
                        loss_idx_value,
                    )
                    experiment_tracker.writer.add_scalar(
                        "Loss/Minibatches", loss.item(), loss_idx_value
                    )
                    experiment_tracker.weight_histograms(mlp.module, loss_idx_value)
                    loss_idx_value = loss_idx_value + 1
                    experiment_tracker.save_state(epoch, i, mlp.module, optimizer)
                dist.barrier()
                torch.cuda.empty_cache()
            if rank == 0:
                avg_loss = total_loss / (dmc.mini_batch + 1)
                log.info(f"Loss after mini-batch {dmc.mini_batch}: {avg_loss}")
                experiment_tracker.writer.add_scalar("Avg Loss/Epochs", avg_loss, epoch)
                experiment_tracker.save_state(epoch, -1, mlp.module, optimizer)
            dist.barrier()
