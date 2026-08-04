from __future__ import annotations

import json

from ml.training.common import finish_validation_only, parse_training_cli, validate_training_inputs
from ml.training.transformer_tasks import train_tamper_localization


def main() -> None:
    config, validate_only = parse_training_cli("Fine-tune SegFormer for possible manipulation localisation")
    validation = validate_training_inputs(config)
    if validate_only:
        finish_validation_only(config, validation)
        return
    print(json.dumps(train_tamper_localization(config), indent=2, default=str))


if __name__ == "__main__":
    main()
