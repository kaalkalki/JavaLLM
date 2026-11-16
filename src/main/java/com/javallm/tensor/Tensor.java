
package com.javallm.tensor;

/**
 * Defines the shape-aware operations provided by tensor implementations.
 *
 * <p>Implementations expose dimensions and strides, indexed access, views,
 * elementwise arithmetic, matrix multiplication, and conversion to row-major
 * values.</p>
 */
public interface Tensor {

    static Tensor of(float[] values, int... dimensions) {
        return DenseTensor.of(values, dimensions);
    }

    int rank();

    int size();

    int dimension(int axis);

    int[] shape();

    int[] strides();

    boolean isContiguous();

    float get(int... indices);

    void set(float value, int... indices);

    Tensor transpose();

    Tensor transpose(int firstAxis, int secondAxis);

    Tensor reshape(int... newShape);

    Tensor add(Tensor other);

    Tensor subtract(Tensor other);

    Tensor multiply(Tensor other);

    Tensor divide(Tensor other);

    Tensor matmul(Tensor other);

    float[] toArray();
}
