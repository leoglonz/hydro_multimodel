import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from injector import inject
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from conf.config import Config
from data.temporally_batched_dataset import TemporallyBatchedDataset
from experiment.experiment_tracker import Tracker
from neural_networks import NeuralNetwork
from physics_models import PhysicsModel

log = logging.getLogger(__name__)


class TestHandler:
    @inject
    def __init__(
        self,
        dataset: TemporallyBatchedDataset,
        neural_networks: Dict[str, NeuralNetwork],
        physics_models: Dict[str, PhysicsModel],
    ):
        self.dataset = dataset
        self.neural_networks = neural_networks
        self.physics_models = physics_models

    def run(self, cfg: Config, experiment_tracker: Tracker):
        rank = cfg.local_rank
        device = cfg.device[rank]

        mlp = self.neural_networks["mlp"]
        streamflow = self.physics_models["streamflow"].eval()
        dmc = self.physics_models["dMC"].eval()
        dmc.epoch = "test"

        # Loading previous model states
        if isinstance(cfg.test.checkpoint, Path):
            file_path = Path(cfg.test.checkpoint)
            log.info(f"Loading from checkpoint: {file_path.stem}")
            state = torch.load(file_path)
            state_dict = state["model_state_dict"]
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(device)
            mlp.load_state_dict(state["model_state_dict"])
        else:
            log.info("Not loading from a checkpoint model.")
            log.info("Initializing new model")
            mlp.initialize_weights()

        # Move NNs to the correct device
        mlp.layers = mlp.layers.to(device)
        mlp = DDP(mlp, device_ids=[device]).eval()

        # We should be using the temporally batched dataset here
        dataloader = DataLoader(
            self.dataset,
            shuffle=cfg.test.shuffle,
            batch_size=cfg.test.batch_size,
            num_workers=0,
            collate_fn=self.dataset.collate_fn,
            drop_last=True,
        )
        warmup = cfg.params.warmup
        gage_indices = self.dataset.hydrofabric.network.gage_information[
            "gage_dict_idx"
        ]
        observations = self.dataset.hydrofabric.observations
        gage_ids = np.array(
            object=[self.dataset.gage_dict["STAID"][x] for x in gage_indices]
        )

        date_time_format = "%Y/%m/%d"
        start = datetime.strptime(self.dataset.dates.start_time, date_time_format)
        current_year = start.year
        start_time = start.strftime("%Y-%m-%d")
        end_time = datetime.strptime(
            self.dataset.dates.end_time, date_time_format
        ).strftime("%Y-%m-%d")
        zones = "_".join(str(zone) for zone in cfg.test.zone)

        observations = observations.streamflow.values

        predictions = np.zeros_like(observations)

        for i, hydrofabric in enumerate(dataloader):
            dmc.mini_batch = i
            dist.barrier()
            with torch.no_grad():
                log.info("Running NN")
                streamflow_predictions = streamflow(hydrofabric=hydrofabric)
                nn_output = mlp(inputs=hydrofabric.normalized_attributes.to(device))
                q_prime = streamflow_predictions["streamflow"] @ hydrofabric.mapping.tm
                dmc_output = dmc(
                    hydrofabric=hydrofabric, parameters=nn_output, streamflow=q_prime
                )

            mini_batch_pred = dmc_output["runoff"].cpu().numpy()
            mini_batch_obs = observations[:, self.dataset.dates.hourly_indices]
            mini_batch_time = self.dataset.dates.hourly_time_range[
                self.dataset.dates.hourly_indices.numpy()
            ]

            predictions[:, self.dataset.dates.hourly_indices] = mini_batch_pred
            dist.barrier()

            # experiment_tracker.set_metrics(
            #     pred=mini_batch_pred[..., warmup:],
            #     target=mini_batch_obs[..., warmup:],
            # )

            # reduced_metrics = self._reduce_metrics(experiment_tracker, device=device)
            # if rank == 0:
            #     x_label = []
            #     data_box = []
            #     for k, v in reduced_metrics.items():
            #         data = v
            #         x_label.append(k)
            #         if data.size > 0:  # Check if data is not empty
            #             data = data[~np.isnan(data)]  # Remove NaNs
            #             if k == "NSE" or k == "KGE":
            #                 data = np.clip(
            #                     data, -1, None
            #                 )  # Clipping the lower bound to -1 for NSE and KGE
            #             data_box.append(data)
            #     experiment_tracker.writer.add_figure(
            #         "Reduced Mini Batch Boxplots",
            #         experiment_tracker.plot_box_fig(
            #             data_box,
            #             x_label,
            #         ),
            #         i,
            #     )
            # dist.barrier()

            _pred_da = xr.DataArray(
                data=mini_batch_pred,
                dims=["gage_ids", "time"],
                coords={"gage_ids": gage_ids, "time": mini_batch_time},
            )
            _obs_da = xr.DataArray(
                data=mini_batch_obs,
                dims=["gage_ids", "time"],
                coords={"gage_ids": gage_ids, "time": mini_batch_time},
            )
            ds = xr.Dataset(
                data_vars={"predictions": _pred_da, "observations": _obs_da},
                attrs={"description": f"Predictions and obs for {current_year}"},
            )
            ds.to_zarr(
                experiment_tracker.zarr_data_path
                / f"zones_{zones}_{current_year}_validation",
                mode="w",
            )
            current_year += 1

        dist.barrier()
        experiment_tracker.set_metrics(
            pred=predictions[..., warmup:],
            target=observations[..., warmup:],
        )
        experiment_tracker.plot_all_time_series(
            predictions[..., warmup:],
            observations[..., warmup:],
            self.dataset.gage_dict,
            self.dataset.dates.hourly_time_range[..., warmup:],
            gage_indices,
            cfg.mode,
        )
        dist.barrier()

        # experiment_tracker.plot_box_metrics(
        #     metrics={
        #         "BIAS": experiment_tracker.metrics.bias,
        #         "RMSE": experiment_tracker.metrics.rmse,
        #         "FLV": experiment_tracker.metrics.flv,
        #         "FHV": experiment_tracker.metrics.fhv,
        #         "NSE": experiment_tracker.metrics.nse,
        #         "KGE": experiment_tracker.metrics.kge,
        #     },
        #     start_time=start_time,
        #     end_time=end_time,
        # )
        # if predictions.shape[0] > 1:
        #     experiment_tracker.plot_cdf(
        #         metric={
        #             "NSE": experiment_tracker.metrics.nse,
        #         },
        #         zones=zones,
        #     )
        experiment_tracker.write_metrics(zones=zones)
        pred_da = xr.DataArray(
            data=predictions,
            dims=["gage_ids", "time"],
            coords={"gage_ids": gage_ids, "time": self.dataset.dates.hourly_time_range},
        )
        obs_da = xr.DataArray(
            data=observations,
            dims=["gage_ids", "time"],
            coords={"gage_ids": gage_ids, "time": self.dataset.dates.hourly_time_range},
        )
        ds = xr.Dataset(
            data_vars={"predictions": pred_da, "observations": obs_da},
            attrs={
                "description": f"Predictions and obs for time period "
                f"{start_time} -"
                f" {end_time}"
            },
        )
        ds.to_zarr(
            experiment_tracker.zarr_data_path
            / f"zones_{zones}_{start_time}_{end_time}_validation",
            mode="w",
        )

    def _reduce_metrics(
        self, experiment_tracker: Tracker, device: int
    ) -> Dict[str, torch.Tensor]:
        bias = experiment_tracker.metrics.bias
        rmse = experiment_tracker.metrics.rmse
        flv = experiment_tracker.metrics.flv
        fhv = experiment_tracker.metrics.fhv
        nse = experiment_tracker.metrics.nse
        kge = experiment_tracker.metrics.kge
        dist.reduce(torch.tensor(bias, device=device), dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(torch.tensor(rmse, device=device), dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(torch.tensor(flv, device=device), dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(torch.tensor(fhv, device=device), dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(torch.tensor(nse, device=device), dst=0, op=dist.ReduceOp.AVG)
        dist.reduce(torch.tensor(kge, device=device), dst=0, op=dist.ReduceOp.AVG)
        metrics = {
            "BIAS": bias,
            "RMSE": rmse,
            "FLV": flv,
            "FHV": fhv,
            "NSE": nse,
            "KGE": kge,
        }
        return metrics
