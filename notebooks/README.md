# Overview of Experimental Parameters

| Exp. | Model  | Type      | Learning Rate | Batch.S. | Epochs | Data     | Task       | Loss     | Act.    | LHC Beta | KL Beta |
|------|--------|-----------|---------------|----------|--------|----------|------------|----------|---------|----------|---------|
| 7A   | NN     | Dense     | 0.0001        | 32       | 12     | EMNIST   | Class.     | LogitBCE | Various |          |         |
| 7B   | AE     | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | Various |          |         |
| 7C   | NN     | Dense     | 0.0001        | 32       | 12     | EMNIST   | Class.     | BCE      | Various |          |         |
| 8A   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | BiCELU  |          |         |
| 8B   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | Id      |          |         |
| 8C   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | Id      |          |         |
| 8D   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | BiCELU  |          |         |
| 8E   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | BiCELU  |          |         |
| 8F   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE      | BiCELU  |          |         |
| 8G   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE+KL   | BiCELU  |          | 0.1     |
| 8H   | PIE/AE | Dense     | 0.0001        | 32       | 12     | EMNIST   | Uns.Enc.   | MSE+KL   | BiCELU  |          | 0.1     |
| 8I   | PIE/AE | Dense     | 0.000075      | 32       | 20     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 8J   | PIE/AE | Sep.Patch | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 8K   | PIE/AE | Dense     | 0.000075      | 32       | 20     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 8L   | PIE/AE | Sep.Patch | 0.00025       | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 8M   | PIE/AE | Sep.Patch | 0.00025       | 32       | 24     | EMNIST   | Gen.Class. | LHC+KL   | BiCELU  | 12       | 96      |
| 8N   | PIE/AE | Sep.Patch | 0.000075      | 32       | 24     | EMNIST   | Gen.Class. | LHC+KL   | BiCELU  | 12       | 48      |
| 8O   | PIE/AE | Sep.Patch | 0.0005        | 32       | 8      | COCO     | Upscaling  | LHC      | BiCELU  | 5        |         |
| 8P   | PIE/AE | Sep.Patch | 0.0005        | 32       | 8      | COCO     | Upscaling  | LHC+KL   | BiCELU  | 16       | 24      |
| 8Q   | PIE/AE | Conv      | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 6        |         |
| 8R   | PIE/AE | Conv      | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 6        |         |
| 8S   | PIE/AE | Sep.Patch | 0.0005        | 32       | 8      | COCO     | Upscaling  | LHC      | BiCELU  | 5        |         |
| 8T   | PIE/AE | Sep.Patch | 0.0005        | 32       | 8      | COCO     | Upscaling  | LHC      | BiCELU  | 5        |         |
| 9A   | IE/IRE | Dense     | 0.000075      | 32       | 20     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 9B   | IE/IRE | Dense     | 0.000075      | 32       | 20     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 9C   | IE/IRE | Sep.Patch | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 9D   | IE/IRE | Sep.Patch | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 9E   | IE     | Dense     | 0.000075      | 32       | 20     | CIFAR100 | Deblur     | LHC      | BiCELU  | 4        |         |
| 9F   | IE     | Sep.Patch | 0.000075      | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 6        |         |
| 9G   | IE     | Sep.Patch | 0.0005        | 32       | 8      | COCO     | Upscaling  | LHC      | BiCELU  | 5        |         |
| 9H   | IE     | Conv      | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 6        |         |
| 9I   | IE     | Conv      | 0.0005        | 32       | 25     | CIFAR100 | Deblur     | LHC      | BiCELU  | 6        |         |

