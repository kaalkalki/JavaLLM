
package com.javallm.nn;

import com.javallm.tensor.Tensor;

import java.util.Objects;

/**
 * Applies the sigmoid-weighted linear unit activation elementwise.
 *
 * <p>For each input value {@code x}, the output is {@code x * sigmoid(x)}. The
 * input shape is preserved.</p>
 */
public final class SiLU implements Layer {

    @Override
    public Tensor forward(Tensor input) {

        Objects.requireNonNull(input, "Input cannot be null");

        float[] inputData = input.toArray();
        float[] outputData = new float[inputData.length];

        for (int index = 0; index < inputData.length; index++) {
            float value = inputData[index];
            float sigmoid = 1.0f / (1.0f + (float) Math.exp(-value));
            outputData[index] = value * sigmoid;
        }

        return Tensor.of(outputData, input.shape());
    }
}
