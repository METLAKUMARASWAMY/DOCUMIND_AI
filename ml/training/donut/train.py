from __future__ import annotations

import json

from ml.training.common import finish_validation_only, parse_training_cli, validate_training_inputs
from ml.training.transformer_tasks import train_donut


def main() -> None:
    config, validate_only = parse_training_cli("Fine-tune Donut for OCR-free document JSON generation")
    validation = validate_training_inputs(config)
    if validate_only:
        finish_validation_only(config, validation)
        return
    print(json.dumps(train_donut(config), indent=2, default=str))


if __name__ == "__main__":
    main()
