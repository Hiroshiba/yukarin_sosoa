import argparse
from pathlib import Path

import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from yukarin_sosoa.config import Config
from yukarin_sosoa.dataset import create_dataset
from yukarin_sosoa.evaluator import Evaluator
from yukarin_sosoa.generator import Generator
from yukarin_sosoa.model import Model, ModelOutput, reduce_result
from yukarin_sosoa.network.predictor import create_predictor
from yukarin_sosoa.utility.pytorch_utility import (
    collate_list,
    detach_cpu,
    init_weights,
    make_optimizer,
    make_scheduler,
    to_device,
)
from yukarin_sosoa.utility.train_utility import Logger, SaveManager


def train(config_yaml_path: Path, output_dir: Path):
    # config
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)
    config = Config.from_dict(config_dict)
    config.add_git_info()

    # dataset
    def _create_loader(dataset, for_train: bool, for_eval: bool):
        batch_size = (
            config.train.eval_batch_size if for_eval else config.train.batch_size
        )
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.train.num_processes,
            collate_fn=collate_list,
            pin_memory=config.train.use_gpu,
            drop_last=for_train,
            timeout=0 if config.train.num_processes == 0 else 30,
            persistent_workers=config.train.num_processes > 0,
        )

    datasets = create_dataset(config.dataset)
    train_loader = _create_loader(datasets["train"], for_train=True, for_eval=False)
    test_loader = _create_loader(datasets["test"], for_train=False, for_eval=False)
    eval_loader = _create_loader(datasets["eval"], for_train=False, for_eval=True)
    valid_loader = _create_loader(datasets["valid"], for_train=False, for_eval=True)

    # predictor
    predictor = create_predictor(config.network)
    device = "cuda" if config.train.use_gpu else "cpu"
    if config.train.pretrained_predictor_path is not None:
        state_dict = torch.load(
            config.train.pretrained_predictor_path, map_location=device
        )
        predictor.load_state_dict(state_dict)
    print("predictor:", predictor)

    # model
    predictor_scripted = torch.jit.script(predictor)
    model = Model(model_config=config.model, predictor=predictor_scripted)
    if config.train.weight_initializer is not None:
        init_weights(model, name=config.train.weight_initializer)
    model.to(device)
    model.train()

    # evaluator
    generator = Generator(
        config=config, predictor=predictor_scripted, use_gpu=config.train.use_gpu
    )
    evaluator = Evaluator(generator=generator)

    # optimizer
    optimizer = make_optimizer(config_dict=config.train.optimizer, model=model)
    scaler = GradScaler(device, enabled=config.train.use_amp)

    # logger
    logger = Logger(
        config_dict=config_dict,
        project_category=config.project.category,
        project_name=config.project.name,
        output_dir=output_dir,
    )

    # snapshot
    snapshot_path = output_dir / "snapshot.pth"
    if not snapshot_path.exists():
        iteration = -1
        epoch = -1
    else:
        snapshot = torch.load(snapshot_path, map_location=device)

        model.load_state_dict(snapshot["model"])
        optimizer.load_state_dict(snapshot["optimizer"])
        scaler.load_state_dict(snapshot["scaler"])
        logger.load_state_dict(snapshot["logger"])

        iteration = snapshot["iteration"]
        epoch = snapshot["epoch"]
        print(f"Loaded snapshot from {snapshot_path} (epoch: {epoch})")

    # scheduler
    scheduler = None
    if config.train.scheduler is not None:
        scheduler = make_scheduler(
            config_dict=config.train.scheduler,
            optimizer=optimizer,
            last_epoch=iteration,
        )

    # save
    save_manager = SaveManager(
        predictor=predictor,
        prefix="predictor_",
        output_dir=output_dir,
        top_num=config.train.model_save_num,
        last_num=config.train.model_save_num,
    )

    output_dir.mkdir(exist_ok=True, parents=True)
    with (output_dir / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # loop
    assert config.train.eval_epoch % config.train.log_epoch == 0
    assert config.train.snapshot_epoch % config.train.eval_epoch == 0

    for _ in range(config.train.stop_epoch):
        epoch += 1
        if epoch > config.train.stop_epoch:
            break

        model.train()

        train_results: list[ModelOutput] = []
        for batch in train_loader:
            iteration += 1

            with autocast(device, enabled=config.train.use_amp):
                batch = to_device(batch, device, non_blocking=True)
                result: ModelOutput = model(batch)

            loss = result["loss"]
            if loss.isnan():
                raise ValueError("loss is NaN")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                scheduler.step()

            train_results.append(detach_cpu(result))

        if epoch % config.train.log_epoch == 0:
            model.eval()

            with torch.inference_mode():
                test_results: list[ModelOutput] = []
                for batch in test_loader:
                    batch = to_device(batch, device, non_blocking=True)
                    result = model(batch)
                    test_results.append(detach_cpu(result))

                summary = {
                    "train": reduce_result(train_results),
                    "test": reduce_result(test_results),
                    "iteration": iteration,
                    "lr": optimizer.param_groups[0]["lr"],
                }

                if epoch % config.train.eval_epoch == 0:
                    eval_results: list[ModelOutput] = []
                    for batch in eval_loader:
                        batch = to_device(batch, device, non_blocking=True)
                        result = evaluator(batch)
                        eval_results.append(detach_cpu(result))
                    summary["eval"] = reduce_result(eval_results)

                    valid_results: list[ModelOutput] = []
                    for batch in valid_loader:
                        batch = to_device(batch, device, non_blocking=True)
                        result = evaluator(batch)
                        valid_results.append(detach_cpu(result))
                    summary["valid"] = reduce_result(valid_results)

                    if epoch % config.train.snapshot_epoch == 0:
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scaler": scaler.state_dict(),
                                "logger": logger.state_dict(),
                                "iteration": iteration,
                                "epoch": epoch,
                            },
                            snapshot_path,
                        )

                        save_manager.save(
                            value=summary["valid"]["value"], step=epoch, judge="min"
                        )

                logger.log(summary=summary, step=epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    train(**vars(parser.parse_args()))
