package com.javallm.model.llama;

import com.javallm.gguf.GgufFile;
import com.javallm.model.LoadedModel;
import com.javallm.model.ModelArchitecture;
import com.javallm.nn.Embedding;
import com.javallm.nn.Linear;
import com.javallm.nn.RMSNorm;
import com.javallm.tensor.Tensor;
import com.javallm.tokenizer.BytePairTokenizer;
import com.javallm.transformer.DenseFeedForward;
import com.javallm.transformer.DecoderModel;
import com.javallm.transformer.FeedForward;
import com.javallm.transformer.LlamaAttention;
import com.javallm.transformer.LlamaTransformerBlock;
import com.javallm.transformer.TransformerBlock;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Builds a Llama decoder and tokenizer from parsed GGUF metadata and tensors.
 *
 * <p>It reads architecture metadata, maps standard Llama tensor names into
 * embeddings, norms, attention projections, and feed-forward projections, then
 * returns those components with the GGUF tokenizer as a {@link LoadedModel}.
 * Unsupported architectures, tensor layouts, or tensor encodings fail with an
 * {@link IOException}.</p>
 */
public final class LlamaArchitecture implements ModelArchitecture {

    private static final String METADATA_ARCHITECTURE_KEY = "general.architecture";
    private static final String LLAMA_ARCHITECTURE = "llama";
    private static final String LLAMA_EMBEDDING_LENGTH_KEY = "llama.embedding_length";
    private static final String LLAMA_FEED_FORWARD_LENGTH_KEY = "llama.feed_forward_length";
    private static final String LLAMA_BLOCK_COUNT_KEY = "llama.block_count";
    private static final String LLAMA_QUERY_HEAD_COUNT_KEY = "llama.attention.head_count";
    private static final String LLAMA_KEY_VALUE_HEAD_COUNT_KEY = "llama.attention.head_count_kv";
    private static final String LLAMA_CONTEXT_LENGTH_KEY = "llama.context_length";
    private static final String LLAMA_VOCABULARY_SIZE_KEY = "llama.vocab_size";
    private static final String LLAMA_ROPE_DIMENSION_COUNT_KEY = "llama.rope.dimension_count";
    private static final String LLAMA_ROPE_FREQUENCY_BASE_KEY = "llama.rope.freq_base";
    private static final String LLAMA_RMS_NORM_EPSILON_KEY = "llama.attention.layer_norm_rms_epsilon";

    private static final Logger LOGGER = Logger.getLogger(LlamaArchitecture.class.getName());

    public LlamaArchitecture() {
    }

    @Override
    public String architectureName() {
        return LLAMA_ARCHITECTURE;
    }

    @Override
    public LoadedModel loadModel(GgufFile gguf) throws IOException {
        Objects.requireNonNull(gguf, "GGUF file cannot be null");
        try {
            Map<String, Object> metadata = gguf.metadata();
            String architecture = stringValue(metadata, METADATA_ARCHITECTURE_KEY);
            if (!architecture.equals(LLAMA_ARCHITECTURE)) {
                throw new IOException("Only the Llama GGUF architecture is supported, received: " + architecture);
            }

            int hiddenSize = positiveInt(metadata, LLAMA_EMBEDDING_LENGTH_KEY);
            int feedForwardSize = positiveInt(metadata, LLAMA_FEED_FORWARD_LENGTH_KEY);
            int layerCount = positiveInt(metadata, LLAMA_BLOCK_COUNT_KEY);
            int queryHeads = positiveInt(metadata, LLAMA_QUERY_HEAD_COUNT_KEY);
            int keyValueHeads = positiveInt(metadata, LLAMA_KEY_VALUE_HEAD_COUNT_KEY);
            int contextLength = positiveInt(metadata, LLAMA_CONTEXT_LENGTH_KEY);
            int vocabularySize = positiveInt(metadata, LLAMA_VOCABULARY_SIZE_KEY);
            int ropeDimensionCount = positiveInt(metadata, LLAMA_ROPE_DIMENSION_COUNT_KEY);
            if (hiddenSize % queryHeads != 0) {
                throw new IOException("Embedding size must be divisible by query head count");
            }
            int headDimension = hiddenSize / queryHeads;
            if (ropeDimensionCount != headDimension) {
                throw new IOException("Partial rotary dimensions are not supported by this loader");
            }
            int keyValueSize = Math.multiplyExact(keyValueHeads, headDimension);
            double ropeTheta = numberValue(metadata, LLAMA_ROPE_FREQUENCY_BASE_KEY);
            float epsilon = (float) numberValue(metadata, LLAMA_RMS_NORM_EPSILON_KEY);

            LOGGER.info(String.format(Locale.ROOT,
                    "Model configuration: layers=%d, hidden=%d, feedForward=%d, queryHeads=%d, keyValueHeads=%d, headDimension=%d, vocabulary=%d, context=%d, ropeTheta=%.0f, rmsEpsilon=%g.",
                    layerCount, hiddenSize, feedForwardSize, queryHeads, keyValueHeads, headDimension,
                    vocabularySize, contextLength, ropeTheta, epsilon));
            Tensor embeddingWeights = gguf.readTensor("token_embd.weight");
            requireShape(embeddingWeights, new int[]{vocabularySize, hiddenSize}, "token_embd.weight");
            Embedding embedding = new Embedding(embeddingWeights);
            ArrayList<TransformerBlock> blocks = new ArrayList<>(layerCount);

            for (int layerIndex = 0; layerIndex < layerCount; layerIndex++) {
                LOGGER.log(Level.FINE, "Loading tensor weights for decoder layer {0}.", layerIndex);
                String prefix = "blk." + layerIndex + ".";
                Tensor attentionNormWeights = gguf.readTensor(prefix + "attn_norm.weight");
                Tensor feedForwardNormWeights = gguf.readTensor(prefix + "ffn_norm.weight");

                Linear query = linear(gguf, prefix + "attn_q");
                Linear key = linear(gguf, prefix + "attn_k");
                Linear value = linear(gguf, prefix + "attn_v");
                Linear attentionOutput = linear(gguf, prefix + "attn_output");
                Linear gate = linear(gguf, prefix + "ffn_gate");
                Linear up = linear(gguf, prefix + "ffn_up");
                Linear down = linear(gguf, prefix + "ffn_down");

                requireMatrix(query.weights(), hiddenSize, hiddenSize, prefix + "attn_q.weight");
                requireMatrix(key.weights(), keyValueSize, hiddenSize, prefix + "attn_k.weight");
                requireMatrix(value.weights(), keyValueSize, hiddenSize, prefix + "attn_v.weight");
                requireMatrix(attentionOutput.weights(), hiddenSize, hiddenSize, prefix + "attn_output.weight");
                requireMatrix(gate.weights(), feedForwardSize, hiddenSize, prefix + "ffn_gate.weight");
                requireMatrix(up.weights(), feedForwardSize, hiddenSize, prefix + "ffn_up.weight");
                requireMatrix(down.weights(), hiddenSize, feedForwardSize, prefix + "ffn_down.weight");

                LlamaAttention attention = new LlamaAttention(
                        query,
                        key,
                        value,
                        attentionOutput,
                        queryHeads,
                        keyValueHeads,
                        ropeTheta
                );
                FeedForward feedForward = new DenseFeedForward(gate, up, down);
                blocks.add(new LlamaTransformerBlock(
                        new RMSNorm(attentionNormWeights, epsilon),
                        attention,
                        new RMSNorm(feedForwardNormWeights, epsilon),
                        feedForward
                ));
                requireSize(attentionNormWeights, hiddenSize, prefix + "attn_norm.weight");
                requireSize(feedForwardNormWeights, hiddenSize, prefix + "ffn_norm.weight");
            }

            Tensor finalNormWeights = gguf.readTensor("output_norm.weight");
            requireSize(finalNormWeights, hiddenSize, "output_norm.weight");
            Tensor outputWeights = gguf.hasTensor("output.weight")
                    ? gguf.readTensor("output.weight")
                    : embeddingWeights;
            requireShape(outputWeights, new int[]{vocabularySize, hiddenSize}, "output.weight");
            BytePairTokenizer tokenizer = new BytePairTokenizer(metadata);
            if (tokenizer.vocabularySize() != vocabularySize) {
                throw new IOException("Tokenizer vocabulary does not match model metadata");
            }
            DecoderModel model = new DecoderModel(
                    embedding,
                    blocks,
                    new RMSNorm(finalNormWeights, epsilon),
                    new Linear(outputWeights)
            );
            LoadedModel loadedModel = new LoadedModel(model, tokenizer, contextLength);
            return loadedModel;
        } catch (IOException exception) {
            throw exception;
        } catch (IllegalArgumentException | ArithmeticException exception) {
            throw new IOException("Could not build Llama model from GGUF: " + exception.getMessage(), exception);
        }
    }

    private static Linear linear(GgufFile gguf, String tensorPrefix) throws IOException {
        Tensor weights = gguf.readTensor(tensorPrefix + ".weight");
        Tensor bias = gguf.hasTensor(tensorPrefix + ".bias") ? gguf.readTensor(tensorPrefix + ".bias") : null;
        return new Linear(weights, bias);
    }

    private static void requireShape(Tensor tensor, int[] expected, String name) throws IOException {
        if (!java.util.Arrays.equals(tensor.shape(), expected)) {
            throw new IOException("Unexpected shape for " + name + ": " + java.util.Arrays.toString(tensor.shape()));
        }
    }

    private static void requireSize(Tensor tensor, int expected, String name) throws IOException {
        if (tensor.rank() != 1 || tensor.dimension(0) != expected) {
            throw new IOException("Unexpected shape for " + name + ": " + java.util.Arrays.toString(tensor.shape()));
        }
    }

    private static void requireMatrix(Tensor tensor, int expectedRows, int expectedColumns, String name) throws IOException {
        if (tensor.rank() != 2 || tensor.dimension(0) != expectedRows || tensor.dimension(1) != expectedColumns) {
            throw new IOException("Unexpected shape for " + name + ": " + java.util.Arrays.toString(tensor.shape()));
        }
    }

    private static String stringValue(Map<String, Object> metadata, String key) throws IOException {
        Object value = metadata.get(key);
        if (value instanceof String string) {
            return string;
        }
        throw new IOException("Missing GGUF string metadata: " + key);
    }

    private static int positiveInt(Map<String, Object> metadata, String key) throws IOException {
        Object value = metadata.get(key);
        if (value instanceof Number number) {
            long result = number.longValue();
            if (result > 0 && result <= Integer.MAX_VALUE) {
                return (int) result;
            }
        }
        throw new IOException("Missing or invalid GGUF integer metadata: " + key);
    }

    private static double numberValue(Map<String, Object> metadata, String key) throws IOException {
        Object value = metadata.get(key);
        if (value instanceof Number number && Double.isFinite(number.doubleValue())) {
            return number.doubleValue();
        }
        throw new IOException("Missing or invalid GGUF numeric metadata: " + key);
    }

}
