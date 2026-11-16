# JavaLLM

A small educational, CPU-only decoder-only LLM inference engine written in Java 25.
It loads Llama-compatible GGUF models, tokenizes prompts with byte-level BPE, and
generates text with temperature, top-k, top-p, and a KV cache.

## Requirements

- Java 25 JDK
- Gradle wrapper (`./gradlew`)
- A compatible GGUF model file

## Compile

```bash
./gradlew compileJava
```

## Download a compatible model

For a small example, download the Apache-2.0 SmolLM2-135M Q4_0 GGUF file from
[QuantFactory/SmolLM2-135M-GGUF](https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF):

```bash
mkdir -p models
curl -L --fail \
  'https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.Q4_0.gguf?download=true' \
  -o models/SmolLM2-135M.Q4_0.gguf
```

Keep downloaded model files outside version control; they are not part of the
source project.

## Generate text

Pass the model path, maximum number of new tokens, and prompt to the Gradle
application task:

```bash
./gradlew run --args='models/SmolLM2-135M.Q4_0.gguf 20 "The meaning of life is"'
```

The command prints the generated continuation. The current CLI uses temperature
0.8, top-k 40, top-p 0.95, and a fixed random seed for repeatable output.

## Inference flow

```text
GGUF file
  -> GGUF reader and model loader
  -> decoder weights + tokenizer

prompt text
  -> tokenizer encodes token IDs
  -> token embeddings
  -> transformer blocks
       RMSNorm -> RoPE causal GQA attention -> residual
       RMSNorm -> SwiGLU feed-forward       -> residual
  -> final RMSNorm and output projection -> next-token logits
  -> temperature / top-k / top-p sampling -> next token
  -> KV cache update -> repeat until EOS or token limit
  -> tokenizer decodes generated token IDs -> continuation text
```

The prompt is evaluated as a prefill sequence. For subsequent generated tokens,
the model evaluates only the new token and reuses cached keys and values from
earlier positions.

## Logging

The engine uses Java Util Logging and writes logs to standard error. The default
`INFO` messages include GGUF model name, architecture, file-type ID, model
dimensions, prompt token count, prefill time, sampling settings, sampled token
IDs, and the generation stop reason. `FINE` messages add prompt token IDs and
incremental cache timings. Prompt text and decoded generated text are not logged.

## Core interfaces

- `Tensor` defines tensor operations; `DenseTensor` is the current CPU
  implementation. `Layer` defines the neural-network `forward` contract.
- `Tokenizer` separates text encoding and decoding from the BPE implementation
  in `BytePairTokenizer`.
- `LanguageModel` defines full-sequence and KV-cache forward calls;
  `DecoderModel` implements it.
- `ModelLoader` loads a `LoadedModel`; `GgufModelLoader` implements the GGUF
  format path.
- `TextGenerator` defines prompt-to-continuation generation;
  `TokenGenerator` implements sampling and cache updates.

## Current compatibility

The loader targets the Llama GGUF architecture and supports F32, F16, Q4_0,
Q4_1, and Q8_0 tensor data. The tokenizer supports GPT-2 byte-level BPE and the
SmolLM digit-splitting pre-tokenizer.
