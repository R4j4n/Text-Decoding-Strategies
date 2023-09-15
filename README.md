

# Text Decoding Strategies in LLMS

This repository provides from-scratch implementations of various text decoding strategies, offering users insights into the inner workings and nuances of each method.

## Table of Contents

- [Text Decoding Strategies in LLMS](#text-decoding-strategies-in-llms)
  - [Table of Contents](#table-of-contents)
  - [Strategies Implemented](#strategies-implemented)
  - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)



## Strategies Implemented

- **Greedy Search**: Picks the most probable next word at each step.
- **Beam Search**: Expands the most promising nodes in a breadth-first manner to maintain a beam of likely sequences.
- **Random Sampling with Temperature**: Selects the next word based on its probability distribution, with temperature controlling the randomness.
- **TOP-K Sampling**: Limits the next word candidates to the top K probable words.
- **TOP-p (nucleus) Sampling**: Chooses the smallest set of words whose cumulative probability exceeds a threshold p.


## Prerequisites

```
plotly==5.16.1
torch==2.0.1
transformers==4.33.1

```


## Usage

For instance:
```python
from greedy import GreedySampler
from beam import BeamSampler
from tempreature import RandomTempSampler
from topk import TOPKsampler
from topp import NucleusSampler
```
For a complete use case, see the [usage.ipynb](https://github.com/R4j4n/Text-Decoding-Strategies/blob/main/usage.ipynb) file.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

