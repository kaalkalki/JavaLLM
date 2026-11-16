
package com.javallm.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * A float tensor backed by a flat array with shape, stride, and offset metadata.
 *
 * <p>Construction copies supplied values. Transpose and reshape can create
 * metadata views over shared storage; arithmetic and matrix multiplication
 * produce new dense tensors.</p>
 */
public final class DenseTensor implements Tensor {

    private final float[] data;
    private final int[] shape;
    private final int[] strides;
    private final int offset;
    private final int size;

    private DenseTensor(
            float[] data,
            int[] shape,
            int[] strides,
            int offset,
            int size
    ) {
        this.data = data;
        this.shape = shape;
        this.strides = strides;
        this.offset = offset;
        this.size = size;
    }

    public static DenseTensor of(float[] values, int... dimensions) {

        Objects.requireNonNull(values, "Values cannot be null");
        Objects.requireNonNull(dimensions, "Dimensions cannot be null");

        int[] shape = dimensions.clone();
        int size = calculateSize(shape);

        if (values.length != size) {
            throw new IllegalArgumentException("Data length " + values.length + " does not match tensor size " + size);
        }

        float[] data = values.clone();
        int[] strides = calculateStrides(shape);

        return new DenseTensor(data, shape, strides, 0, size);
    }

    @Override
    public int rank() {
        return shape.length;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public int dimension(int axis) {

        if (axis < 0 || axis >= rank()) {
            throw new IndexOutOfBoundsException("Invalid axis: " + axis);
        }

        return shape[axis];
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    @Override
    public boolean isContiguous() {

        if (size == 0) {
            return true;
        }

        int expectedStride = 1;

        for (int axis = rank() - 1; axis >= 0; axis--) {
            if (shape[axis] == 1) {
                continue;
            }
            if (strides[axis] != expectedStride) {
                return false;
            }
            expectedStride *= shape[axis];
        }
        return true;
    }

    @Override
    public float get(int... indices) {
        int dataIndex = calculateIndex(indices);
        return data[dataIndex];
    }

    @Override
    public void set(float value, int... indices) {
        int dataIndex = calculateIndex(indices);
        data[dataIndex] = value;
    }

    @Override
    public Tensor transpose() {
        if (rank() != 2) {
            throw new IllegalStateException("Default transpose requires a rank-2 tensor");
        }
        return transpose(0, 1);
    }

    @Override
    public Tensor transpose(int firstAxis, int secondAxis) {
        if (firstAxis < 0 || firstAxis >= rank()) {
            throw new IndexOutOfBoundsException("Invalid first axis: " + firstAxis);
        }

        if (secondAxis < 0 || secondAxis >= rank()) {
            throw new IndexOutOfBoundsException("Invalid second axis: " + secondAxis);
        }

        int[] transposedShape = shape.clone();
        int[] transposedStrides = strides.clone();

        int temporaryDimension = transposedShape[firstAxis];
        transposedShape[firstAxis] = transposedShape[secondAxis];
        transposedShape[secondAxis] = temporaryDimension;

        int temporaryStride = transposedStrides[firstAxis];
        transposedStrides[firstAxis] = transposedStrides[secondAxis];
        transposedStrides[secondAxis] = temporaryStride;

        return new DenseTensor(
                data,
                transposedShape,
                transposedStrides,
                offset,
                size
        );
    }

    @Override
    public Tensor reshape(int... newShape) {
        Objects.requireNonNull(newShape, "New shape cannot be null");
        int[] reshapedShape = newShape.clone();
        int reshapedSize = calculateSize(reshapedShape);
        if (reshapedSize != size) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + size + " into shape " + Arrays.toString(reshapedShape));
        }

        if (!isContiguous()) {
            throw new IllegalStateException("Cannot reshape a non-contiguous tensor");
        }

        int[] reshapedStrides = calculateStrides(reshapedShape);
        return new DenseTensor(
                data,
                reshapedShape,
                reshapedStrides,
                offset,
                size
        );
    }

    @Override
    public Tensor add(Tensor other) {
        return apply(other, Operation.ADD);
    }

    @Override
    public Tensor subtract(Tensor other) {
        return apply(other, Operation.SUBTRACT);
    }

    @Override
    public Tensor multiply(Tensor other) {
        return apply(other, Operation.MULTIPLY);
    }

    @Override
    public Tensor divide(Tensor other) {
        return apply(other, Operation.DIVIDE);
    }

    private Tensor apply(Tensor other, Operation operation) {
        Objects.requireNonNull(other, "Other tensor cannot be null");
        int[] otherShape = other.shape();
        int[] outputShape = calculateBroadcastShape(shape, otherShape);

        int outputSize = calculateSize(outputShape);
        int[] outputStrides = calculateStrides(outputShape);

        float[] outputData = new float[outputSize];

        for (int linearIndex = 0; linearIndex < outputSize; linearIndex++) {
            int[] outputCoordinates = calculateCoordinates(linearIndex, outputShape);

            int[] firstCoordinates = mapBroadcastCoordinates(outputCoordinates, shape);
            int[] secondCoordinates = mapBroadcastCoordinates(outputCoordinates, otherShape);

            float firstValue = get(firstCoordinates);
            float secondValue = other.get(secondCoordinates);

            float result;

            switch (operation) {
                case ADD -> result = firstValue + secondValue;
                case SUBTRACT -> result = firstValue - secondValue;
                case MULTIPLY -> result = firstValue * secondValue;
                case DIVIDE -> result = firstValue / secondValue;
                default -> throw new IllegalStateException("Unsupported operation: " + operation);
            }

            outputData[linearIndex] = result;
        }

        return new DenseTensor(
                outputData,
                outputShape,
                outputStrides,
                0,
                outputSize
        );
    }

    @Override
    public Tensor matmul(Tensor other) {
        Objects.requireNonNull(other, "Other tensor cannot be null");
        if (rank() < 2 || other.rank() < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires tensors with rank >= 2");
        }

        int rows = shape[rank() - 2];
        int sharedDimension = shape[rank() - 1];

        int otherSharedDimension = other.dimension(other.rank() - 2);
        int columns = other.dimension(other.rank() - 1);

        if (sharedDimension != otherSharedDimension) {
            throw new IllegalArgumentException("Incompatible matrix dimensions: " + rows + "x" + sharedDimension + " and " + otherSharedDimension + "x" + columns);
        }

        int[] firstBatchShape = Arrays.copyOf(shape, rank() - 2);
        int[] otherShape = other.shape();
        int[] secondBatchShape = Arrays.copyOf(otherShape, other.rank() - 2);
        int[] outputBatchShape = calculateBroadcastShape(firstBatchShape, secondBatchShape);
        int[] outputShape = Arrays.copyOf(outputBatchShape, outputBatchShape.length + 2);

        outputShape[outputShape.length - 2] = rows;
        outputShape[outputShape.length - 1] = columns;

        int batchSize = calculateSize(outputBatchShape);

        int outputSize;
        try {
            outputSize = Math.multiplyExact(batchSize, Math.multiplyExact(rows, columns));
        } catch (ArithmeticException exception) {
            throw new IllegalArgumentException("Matrix result exceeds supported integer range", exception);
        }

        float[] outputData = new float[outputSize];

        for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            int[] batchCoordinates = calculateCoordinates(batchIndex, outputBatchShape);
            int[] firstBatchCoordinates = mapBroadcastCoordinates(batchCoordinates, firstBatchShape);
            int[] secondBatchCoordinates = mapBroadcastCoordinates(batchCoordinates, secondBatchShape);

            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    float sum = 0.0f;
                    for (int k = 0; k < sharedDimension; k++) {
                        int[] firstCoordinates = Arrays.copyOf(firstBatchCoordinates, firstBatchCoordinates.length + 2);
                        firstCoordinates[firstCoordinates.length - 2] = row;
                        firstCoordinates[firstCoordinates.length - 1] = k;

                        int[] secondCoordinates = Arrays.copyOf(secondBatchCoordinates, secondBatchCoordinates.length + 2);
                        secondCoordinates[secondCoordinates.length - 2] = k;
                        secondCoordinates[secondCoordinates.length - 1] = column;

                        float firstValue = get(firstCoordinates);
                        float secondValue = other.get(secondCoordinates);
                        sum += firstValue * secondValue;
                    }

                    int outputIndex;
                    try {
                        outputIndex = Math.addExact(
                                Math.multiplyExact(batchIndex, Math.multiplyExact(rows, columns)),
                                Math.addExact(Math.multiplyExact(row, columns), column)
                        );
                    } catch (ArithmeticException exception) {
                        throw new IllegalArgumentException("Matrix output index exceeds supported integer range", exception);
                    }
                    outputData[outputIndex] = sum;
                }
            }
        }

        return new DenseTensor(
                outputData,
                outputShape,
                calculateStrides(outputShape),
                0,
                outputSize
        );
    }

    @Override
    public float[] toArray() {
        float[] result = new float[size];
        for (int linearIndex = 0; linearIndex < size; linearIndex++) {
            int[] coordinates = calculateCoordinates(linearIndex, shape);
            result[linearIndex] = get(coordinates);
        }
        return result;
    }

    private int calculateIndex(int[] indices) {
        Objects.requireNonNull(indices, "Indices cannot be null");
        if (indices.length != rank()) {
            throw new IllegalArgumentException("Expected " + rank() + " indices, received " + indices.length);
        }

        long dataIndex = offset;
        try {
            for (int axis = 0; axis < rank(); axis++) {
                int coordinate = indices[axis];
                if (coordinate < 0 || coordinate >= shape[axis]) {
                    throw new IndexOutOfBoundsException("Index " + coordinate + " is out of bounds for axis " + axis + " with dimension " + shape[axis]);
                }
                dataIndex = Math.addExact(dataIndex, Math.multiplyExact((long) coordinate, strides[axis]));
            }
        } catch (ArithmeticException exception) {
            throw new IndexOutOfBoundsException("Calculated data index exceeds supported integer range");
        }
        if (dataIndex < 0 || dataIndex >= data.length) {
            throw new IndexOutOfBoundsException("Calculated data index is outside the backing array: " + dataIndex);
        }
        return (int) dataIndex;
    }

    private static int[] calculateBroadcastShape(int[] firstShape, int[] secondShape) {
        int outputRank = Math.max(firstShape.length, secondShape.length);
        int[] outputShape = new int[outputRank];
        for (int axis = 0; axis < outputRank; axis++) {
            int firstAxis = axis - (outputRank - firstShape.length);
            int secondAxis = axis - (outputRank - secondShape.length);

            int firstDimension = firstAxis < 0 ? 1 : firstShape[firstAxis];
            int secondDimension = secondAxis < 0 ? 1 : secondShape[secondAxis];

            if (firstDimension != secondDimension && firstDimension != 1 && secondDimension != 1) {
                throw new IllegalArgumentException("Incompatible shapes for broadcasting: " + Arrays.toString(firstShape) + " and " + Arrays.toString(secondShape));
            }

            outputShape[axis] = firstDimension == 1 ? secondDimension : firstDimension;
        }

        return outputShape;
    }

    private static int[] mapBroadcastCoordinates(int[] outputCoordinates, int[] inputShape) {
        int[] inputCoordinates = new int[inputShape.length];
        int rankDifference = outputCoordinates.length - inputShape.length;
        for (int axis = 0; axis < inputShape.length; axis++) {
            inputCoordinates[axis] = (inputShape[axis] != 1) ? outputCoordinates[axis + rankDifference] : 0;
        }
        return inputCoordinates;
    }

    private static int[] calculateCoordinates(int linearIndex, int[] dimensions) {
        int size = calculateSize(dimensions);
        if (linearIndex < 0 || linearIndex >= size) {
            throw new IndexOutOfBoundsException("Linear index " + linearIndex + " is outside tensor size " + size);
        }
        int[] coordinates = new int[dimensions.length];
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            coordinates[axis] = linearIndex % dimensions[axis];
            linearIndex /= dimensions[axis];
        }
        return coordinates;
    }

    private static int calculateSize(int[] dimensions) {
        boolean hasZeroDimension = false;
        for (int dimension : dimensions) {
            if (dimension < 0) {
                throw new IllegalArgumentException("Negative dimension: " + dimension);
            }
            hasZeroDimension |= dimension == 0;
        }
        if (hasZeroDimension) {
            return 0;
        }

        int size = 1;
        for (int axis = 0; axis < dimensions.length; axis++) {
            try {
                size = Math.multiplyExact(size, dimensions[axis]);
            } catch (ArithmeticException exception) {
                throw new IllegalArgumentException("Tensor size exceeds supported integer range", exception);
            }
        }

        return size;
    }

    private static int[] calculateStrides(int[] dimensions) {
        int[] strides = new int[dimensions.length];
        long stride = 1L;
        boolean overflowed = false;
        boolean empty = calculateSize(dimensions) == 0;
        for (int axis = dimensions.length - 1; axis >= 0; axis--) {
            if (overflowed || stride > Integer.MAX_VALUE) {
                if (!empty) {
                    throw new IllegalArgumentException("Tensor strides exceed supported integer range");
                }
                strides[axis] = 0;
            } else {
                strides[axis] = (int) stride;
            }

            if (dimensions[axis] == 0) {
                stride = 0L;
                overflowed = false;
            } else if (!overflowed) {
                try {
                    stride = Math.multiplyExact(stride, dimensions[axis]);
                } catch (ArithmeticException exception) {
                    if (!empty) {
                        throw new IllegalArgumentException("Tensor strides exceed supported integer range", exception);
                    }
                    overflowed = true;
                }
            }
        }
        return strides;
    }

    @Override
    public String toString() {

        return "DenseTensor{" +
                "shape=" + Arrays.toString(shape) +
                ", strides=" + Arrays.toString(strides) +
                ", offset=" + offset +
                ", size=" + size +
                ", data=" + Arrays.toString(toArray()) +
                '}';
    }

    /** Elementwise operations supported by {@link #apply(Tensor, Operation)}. */
    private enum Operation {
        ADD,
        SUBTRACT,
        MULTIPLY,
        DIVIDE
    }
}
