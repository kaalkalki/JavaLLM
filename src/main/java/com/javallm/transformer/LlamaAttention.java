package com.javallm.transformer;

import com.javallm.nn.Linear;
import com.javallm.nn.RoPE;
import com.javallm.tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;
import java.util.logging.Logger;

/**
 * Llama causal attention for {@code [batch, sequence, hidden]} inputs.
 *
 * <p>Query, key, and value projections are split into heads; RoPE is applied to
 * query and key heads, then each query attends only to current or earlier
 * positions. Multiple query heads can share each key/value head. The output
 * projection restores the hidden feature size.</p>
 */
public final class LlamaAttention implements Attention {

    private static final Logger LOGGER = Logger.getLogger(LlamaAttention.class.getName());

    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;
    private final Linear outputProjection;
    private final int queryHeads;
    private final int keyValueHeads;
    private final int headDimension;
    private final int hiddenSize;
    private final RoPE rope;

    public LlamaAttention(
            Linear queryProjection,
            Linear keyProjection,
            Linear valueProjection,
            Linear outputProjection,
            int queryHeads,
            int keyValueHeads,
            double ropeTheta
    ) {
        this.queryProjection = Objects.requireNonNull(queryProjection, "Query projection cannot be null");
        this.keyProjection = Objects.requireNonNull(keyProjection, "Key projection cannot be null");
        this.valueProjection = Objects.requireNonNull(valueProjection, "Value projection cannot be null");
        this.outputProjection = Objects.requireNonNull(outputProjection, "Output projection cannot be null");
        if (queryHeads <= 0 || keyValueHeads <= 0 || queryHeads % keyValueHeads != 0) {
            throw new IllegalArgumentException("Query heads must be a positive multiple of key/value heads");
        }

        Tensor queryWeights = queryProjection.weights();
        Tensor keyWeights = keyProjection.weights();
        Tensor valueWeights = valueProjection.weights();
        Tensor outputWeights = outputProjection.weights();
        if (queryWeights.rank() != 2 || keyWeights.rank() != 2 || valueWeights.rank() != 2 || outputWeights.rank() != 2) {
            throw new IllegalArgumentException("Attention projection weights must be rank-2 tensors");
        }
        if (queryWeights.dimension(0) % queryHeads != 0) {
            throw new IllegalArgumentException("Query projection size must be divisible by query heads");
        }

        this.queryHeads = queryHeads;
        this.keyValueHeads = keyValueHeads;
        this.headDimension = queryWeights.dimension(0) / queryHeads;
        this.hiddenSize = queryWeights.dimension(1);
        if (headDimension <= 0 || headDimension % 2 != 0) {
            throw new IllegalArgumentException("Attention head dimension must be positive and even");
        }
        int keyValueWidth;
        try {
            keyValueWidth = Math.multiplyExact(keyValueHeads, headDimension);
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException("Key/value projection size exceeds supported range", exception);
        }
        if (keyWeights.dimension(0) != keyValueWidth
                || valueWeights.dimension(0) != keyValueWidth
                || keyWeights.dimension(1) != hiddenSize
                || valueWeights.dimension(1) != hiddenSize
                || outputWeights.dimension(0) != hiddenSize
                || outputWeights.dimension(1) != hiddenSize) {
            throw new IllegalArgumentException("Attention projection dimensions are inconsistent");
        }
        this.rope = new RoPE(ropeTheta);
    }

    @Override
    public int hiddenSize() {
        return hiddenSize;
    }

    public int keyValueHeads() {
        return keyValueHeads;
    }

    @Override
    public int cacheHeadCount() {
        return keyValueHeads;
    }

    public int headDimension() {
        return headDimension;
    }

    @Override
    public int cacheHeadDimension() {
        return headDimension;
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, 0, null, -1);
    }

    @Override
    public Tensor forwardWithCache(Tensor input, int startPosition, KeyValueCache cache, int layerIndex) {
        Objects.requireNonNull(input, "Input cannot be null");
        if (input.rank() != 3 || input.dimension(2) != hiddenSize) {
            throw new IllegalArgumentException("Attention input must have shape [batch, sequence, hidden]");
        }

        int batchSize = input.dimension(0);
        int sequenceLength = input.dimension(1);
        if (batchSize <= 0 || sequenceLength <= 0) {
            throw new IllegalArgumentException("Attention batch and sequence dimensions must be positive");
        }
        if (startPosition < 0 || (cache == null && startPosition != 0)) {
            throw new IllegalArgumentException("Invalid attention start position");
        }
        if (cache != null && (batchSize != 1 || startPosition != cache.sequenceLength())) {
            throw new IllegalArgumentException("Cached attention requires one batch and the cache's next position");
        }

        boolean logFirstLayer = layerIndex == 0;
        if (logFirstLayer) {
            LOGGER.info("SELF ATTENTION layer 0: input shape=" + Arrays.toString(input.shape()));
        }

        Tensor queryProjectionOutput = queryProjection.forward(input);
        Tensor keyProjectionOutput = keyProjection.forward(input);
        Tensor valueProjectionOutput = valueProjection.forward(input);
        if (logFirstLayer) {
            logTensor("Q projection", queryProjectionOutput);
            logTensor("K projection", keyProjectionOutput);
            logTensor("V projection", valueProjectionOutput);
        }

        Tensor query = splitHeads(queryProjectionOutput, queryHeads, batchSize, sequenceLength);
        Tensor key = splitHeads(keyProjectionOutput, keyValueHeads, batchSize, sequenceLength);
        Tensor value = splitHeads(valueProjectionOutput, keyValueHeads, batchSize, sequenceLength);
        if (logFirstLayer) {
            LOGGER.info("SELF ATTENTION layer 0: Q shape after head split=" + Arrays.toString(query.shape()));
            LOGGER.info("SELF ATTENTION layer 0: K shape after head split=" + Arrays.toString(key.shape()));
        }
        query = rope.forward(query, startPosition);
        key = rope.forward(key, startPosition);
        if (logFirstLayer) {
            LOGGER.info("SELF ATTENTION layer 0: Q shape after RoPE=" + Arrays.toString(query.shape()));
            LOGGER.info("SELF ATTENTION layer 0: K shape after RoPE=" + Arrays.toString(key.shape()));
        }
        if (cache != null) {
            int cacheLengthBeforeAppend = cache.layerSequenceLength(layerIndex);
            if (logFirstLayer) {
                LOGGER.info("SELF ATTENTION layer 0: KV layer cache length before append=" + cacheLengthBeforeAppend);
            }
            cache.append(layerIndex, key, value, startPosition);
            if (logFirstLayer) {
                LOGGER.info("SELF ATTENTION layer 0: KV layer cache length after append="
                        + cache.layerSequenceLength(layerIndex));
            }
        }

        float[] queryData = query.toArray();
        float[] keyData = key.toArray();
        float[] valueData = value.toArray();
        float[] contextData = new float[batchSize * sequenceLength * hiddenSize];
        double scale = 1.0 / Math.sqrt(headDimension);
        int headsPerKeyValueHead = queryHeads / keyValueHeads;

        for (int batch = 0; batch < batchSize; batch++) {
            for (int head = 0; head < queryHeads; head++) {
                int keyValueHead = head / headsPerKeyValueHead;
                for (int queryPosition = 0; queryPosition < sequenceLength; queryPosition++) {
                    int lastKeyPosition = startPosition + queryPosition;
                    float[] scores = new float[lastKeyPosition + 1];
                    float maximumScore = Float.NEGATIVE_INFINITY;
                    for (int keyPosition = 0; keyPosition <= lastKeyPosition; keyPosition++) {
                        float score = 0.0f;
                        for (int dimension = 0; dimension < headDimension; dimension++) {
                            int queryIndex = headIndex(batch, head, queryPosition, dimension, queryHeads, sequenceLength, headDimension);
                            float keyValue = cache == null
                                    ? keyData[headIndex(batch, keyValueHead, keyPosition, dimension, keyValueHeads, sequenceLength, headDimension)]
                                    : cache.key(layerIndex, keyValueHead, keyPosition, dimension);
                            score += queryData[queryIndex] * keyValue;
                        }
                        scores[keyPosition] = (float) (score * scale);
                        maximumScore = Math.max(maximumScore, scores[keyPosition]);
                    }

                    float scoreSum = 0.0f;
                    for (int keyPosition = 0; keyPosition <= lastKeyPosition; keyPosition++) {
                        scores[keyPosition] = (float) Math.exp(scores[keyPosition] - maximumScore);
                        scoreSum += scores[keyPosition];
                    }

                    for (int dimension = 0; dimension < headDimension; dimension++) {
                        float result = 0.0f;
                        for (int keyPosition = 0; keyPosition <= lastKeyPosition; keyPosition++) {
                            float valueValue = cache == null
                                    ? valueData[headIndex(batch, keyValueHead, keyPosition, dimension, keyValueHeads, sequenceLength, headDimension)]
                                    : cache.value(layerIndex, keyValueHead, keyPosition, dimension);
                            result += (scores[keyPosition] / scoreSum) * valueValue;
                        }
                        int contextIndex = (batch * sequenceLength + queryPosition) * hiddenSize + head * headDimension + dimension;
                        contextData[contextIndex] = result;
                    }
                }
            }
        }

        Tensor context = Tensor.of(contextData, batchSize, sequenceLength, hiddenSize);
        Tensor attentionOutput = outputProjection.forward(context);
        if (logFirstLayer) {
            logTensor("attention output", attentionOutput);
        }
        return attentionOutput;
    }

    /** Compatibility overload retained for callers of the former SelfAttention API. */
    public Tensor forward(Tensor input, int startPosition, KeyValueCache cache, int layerIndex) {
        return forwardWithCache(input, startPosition, cache, layerIndex);
    }

    private static void logTensor(String label, Tensor tensor) {
        float[] values = tensor.toArray();
        float[] prefix = Arrays.copyOf(values, Math.min(10, values.length));
        LOGGER.info("SELF ATTENTION layer 0: " + label + " shape=" + Arrays.toString(tensor.shape())
                + ", first values=" + Arrays.toString(prefix));
    }

    private static Tensor splitHeads(Tensor projected, int heads, int batchSize, int sequenceLength) {
        int headDimension = projected.dimension(2) / heads;
        float[] source = projected.toArray();
        float[] result = new float[source.length];
        for (int batch = 0; batch < batchSize; batch++) {
            for (int sequence = 0; sequence < sequenceLength; sequence++) {
                for (int head = 0; head < heads; head++) {
                    for (int dimension = 0; dimension < headDimension; dimension++) {
                        int sourceIndex = (batch * sequenceLength + sequence) * heads * headDimension + head * headDimension + dimension;
                        int targetIndex = headIndex(batch, head, sequence, dimension, heads, sequenceLength, headDimension);
                        result[targetIndex] = source[sourceIndex];
                    }
                }
            }
        }
        return Tensor.of(result, batchSize, heads, sequenceLength, headDimension);
    }

    private static int headIndex(int batch, int head, int sequence, int dimension, int heads, int sequenceLength, int headDimension) {
        return ((batch * heads + head) * sequenceLength + sequence) * headDimension + dimension;
    }
}
