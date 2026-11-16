package com.javallm.transformer;

import com.javallm.tensor.Tensor;

import java.util.Arrays;

/**
 * Dynamically growing, single-sequence cache for decoder key and value states.
 *
 * <p>Each transformer layer stores its own key/value vectors as tokens are
 * processed. The cache tracks the shared sequence position and enforces the
 * configured maximum context length.</p>
 */
public final class KeyValueCache {

    private final int layerCount;
    private final int keyValueHeads;
    private final int headDimension;
    private final int maximumSequenceLength;
    private final LayerCache[] layers;
    private int sequenceLength;

    public KeyValueCache(int layerCount, int keyValueHeads, int headDimension, int maximumSequenceLength) {
        if (layerCount < 0 || keyValueHeads <= 0 || headDimension <= 0 || maximumSequenceLength <= 0) {
            throw new IllegalArgumentException("Invalid KV cache dimensions");
        }
        this.layerCount = layerCount;
        this.keyValueHeads = keyValueHeads;
        this.headDimension = headDimension;
        this.maximumSequenceLength = maximumSequenceLength;
        this.layers = new LayerCache[layerCount];
        for (int layer = 0; layer < layerCount; layer++) {
            layers[layer] = new LayerCache(keyValueHeads);
        }
    }

    public int sequenceLength() {
        return sequenceLength;
    }

    public int maximumSequenceLength() {
        return maximumSequenceLength;
    }

    int layerSequenceLength(int layerIndex) {
        if (layerIndex < 0 || layerIndex >= layerCount) {
            throw new IndexOutOfBoundsException("Invalid KV cache layer: " + layerIndex);
        }
        return layers[layerIndex].length;
    }

    public void reset() {
        for (LayerCache layer : layers) {
            layer.length = 0;
        }
        sequenceLength = 0;
    }

    void append(int layerIndex, Tensor keys, Tensor values, int startPosition) {
        if (layerIndex < 0 || layerIndex >= layerCount) {
            throw new IndexOutOfBoundsException("Invalid KV cache layer: " + layerIndex);
        }
        if (keys.rank() != 4 || values.rank() != 4
                || keys.dimension(0) != 1 || values.dimension(0) != 1
                || keys.dimension(1) != keyValueHeads || values.dimension(1) != keyValueHeads
                || keys.dimension(2) != values.dimension(2)
                || keys.dimension(3) != headDimension || values.dimension(3) != headDimension) {
            throw new IllegalArgumentException("KV cache tensors have incompatible shapes");
        }
        LayerCache layer = layers[layerIndex];
        if (startPosition != layer.length) {
            throw new IllegalStateException("KV cache layer is at " + layer.length + ", requested position " + startPosition);
        }
        int tokenCount = keys.dimension(2);
        int newLength = Math.addExact(startPosition, tokenCount);
        if (newLength > maximumSequenceLength) {
            throw new IllegalArgumentException("KV cache exceeds configured context length");
        }
        ensureCapacity(layer, newLength);

        float[] keyValues = keys.toArray();
        float[] valueValues = values.toArray();
        for (int head = 0; head < keyValueHeads; head++) {
            for (int token = 0; token < tokenCount; token++) {
                int sourceOffset = (head * tokenCount + token) * headDimension;
                int targetOffset = (startPosition + token) * headDimension;
                System.arraycopy(keyValues, sourceOffset, layer.keys[head], targetOffset, headDimension);
                System.arraycopy(valueValues, sourceOffset, layer.values[head], targetOffset, headDimension);
            }
        }
        layer.length = newLength;
    }

    float key(int layerIndex, int head, int position, int dimension) {
        return layers[layerIndex].keys[head][position * headDimension + dimension];
    }

    float value(int layerIndex, int head, int position, int dimension) {
        return layers[layerIndex].values[head][position * headDimension + dimension];
    }

    void completePrefill(int newSequenceLength) {
        if (newSequenceLength < 0 || newSequenceLength > maximumSequenceLength) {
            throw new IllegalArgumentException("Invalid KV cache sequence length");
        }
        for (int layerIndex = 0; layerIndex < layers.length; layerIndex++) {
            if (layers[layerIndex].length != newSequenceLength) {
                throw new IllegalStateException("KV cache layer " + layerIndex + " did not reach expected sequence length");
            }
        }
        sequenceLength = newSequenceLength;
    }

    private void ensureCapacity(LayerCache layer, int requiredLength) {
        if (requiredLength <= layer.capacity) {
            return;
        }
        int newCapacity = Math.max(16, layer.capacity);
        while (newCapacity < requiredLength) {
            int doubled = newCapacity > maximumSequenceLength / 2 ? maximumSequenceLength : newCapacity * 2;
            if (doubled <= newCapacity) {
                newCapacity = requiredLength;
                break;
            }
            newCapacity = doubled;
        }
        int newArrayLength = Math.multiplyExact(newCapacity, headDimension);
        for (int head = 0; head < keyValueHeads; head++) {
            layer.keys[head] = Arrays.copyOf(layer.keys[head], newArrayLength);
            layer.values[head] = Arrays.copyOf(layer.values[head], newArrayLength);
        }
        layer.capacity = newCapacity;
    }

    /** Stores cached keys and values for one transformer layer. */
    private static final class LayerCache {
        private final float[][] keys;
        private final float[][] values;
        private int capacity;
        private int length;

        private LayerCache(int keyValueHeads) {
            this.keys = new float[keyValueHeads][];
            this.values = new float[keyValueHeads][];
            for (int head = 0; head < keyValueHeads; head++) {
                keys[head] = new float[0];
                values[head] = new float[0];
            }
        }
    }
}
