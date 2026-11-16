package com.javallm.gguf;

import com.javallm.tensor.Tensor;
import java.io.Closeable;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Reads GGUF headers, metadata, tensor descriptors, and supported CPU tensor formats.
 *
 * <p>The reader owns an open file handle and must be closed after use. Metadata
 * remains available as Java values, while tensor payloads are read on demand and
 * converted to the project's dense float tensor representation.</p>
 */
public final class GgufFile implements Closeable {

    private static final String METADATA_ALIGNMENT_KEY = "general.alignment";
    private static final long MAX_METADATA_ENTRIES = 1_000_000L;
    private static final long MAX_ARRAY_ENTRIES = 10_000_000L;

    private final RandomAccessFile file;
    private final Map<String, Object> metadata;
    private final Map<String, TensorInfo> tensors;
    private final long tensorDataOffset;
    private final int alignment;

    private GgufFile(Path path) throws IOException {
        this.file = new RandomAccessFile(path.toFile(), "r");
        this.metadata = new LinkedHashMap<>();
        this.tensors = new LinkedHashMap<>();

        byte[] magic = new byte[4];
        file.readFully(magic);
        if (magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
            throw new IOException("File does not have a GGUF header");
        }
        int version = readIntLE();
        if (version < 2 || version > 3) {
            throw new IOException("Unsupported GGUF version: " + version);
        }

        long tensorCount = readLongLE();
        long metadataCount = readLongLE();
        requireCount(tensorCount, MAX_METADATA_ENTRIES, "tensor");
        requireCount(metadataCount, MAX_METADATA_ENTRIES, "metadata");

        for (long index = 0; index < metadataCount; index++) {
            String key = readString();
            int type = readIntLE();
            metadata.put(key, readValue(type));
        }

        for (long index = 0; index < tensorCount; index++) {
            String name = readString();
            int dimensionCount = readIntLE();
            if (dimensionCount < 1 || dimensionCount > 4) {
                throw new IOException("Unsupported tensor rank for " + name + ": " + dimensionCount);
            }
            int[] shape = new int[dimensionCount];
            for (int dimension = 0; dimension < dimensionCount; dimension++) {
                long value = readLongLE();
                if (value < 0 || value > Integer.MAX_VALUE) {
                    throw new IOException("Tensor dimension is outside supported range: " + name);
                }
                shape[dimensionCount - dimension - 1] = (int) value;
            }
            int type = readIntLE();
            long offset = readLongLE();
            if (offset < 0) {
                throw new IOException("Tensor offset exceeds supported range: " + name);
            }
            tensors.put(name, new TensorInfo(name, shape, type, offset));
        }

        Object alignmentValue = metadata.get(METADATA_ALIGNMENT_KEY);
        this.alignment = alignmentValue instanceof Number number ? number.intValue() : 32;
        if (alignment <= 0) {
            throw new IOException("Invalid GGUF tensor alignment: " + alignment);
        }
        long position = file.getFilePointer();
        this.tensorDataOffset = Math.multiplyExact((position + alignment - 1) / alignment, alignment);
    }

    public static GgufFile open(Path path) throws IOException {
        Objects.requireNonNull(path, "Path cannot be null");
        return new GgufFile(path);
    }

    public Map<String, Object> metadata() {
        return Collections.unmodifiableMap(metadata);
    }

    public Object metadata(String key) {
        return metadata.get(key);
    }

    public Map<String, int[]> tensorShapes() {
        Map<String, int[]> shapes = new LinkedHashMap<>();
        for (TensorInfo info : tensors.values()) {
            shapes.put(info.name(), info.shape().clone());
        }
        return Collections.unmodifiableMap(shapes);
    }

    public boolean hasTensor(String name) {
        return tensors.containsKey(name);
    }

    public Tensor readTensor(String name) throws IOException {
        TensorInfo info = tensors.get(name);
        if (info == null) {
            throw new IOException("GGUF tensor not found: " + name);
        }
        int size = tensorSize(info.shape());
        long absoluteOffset;
        try {
            absoluteOffset = Math.addExact(tensorDataOffset, info.offset());
        } catch (ArithmeticException exception) {
            throw new IOException("Tensor offset exceeds supported range: " + name, exception);
        }
        file.seek(absoluteOffset);
        float[] values = switch (info.type()) {
            case 0 -> readF32(size);
            case 1 -> readF16(size);
            case 2 -> readQ4_0(size);
            case 3 -> readQ4_1(size);
            case 8 -> readQ8_0(size);
            default -> throw new IOException("Unsupported GGUF tensor type " + info.type() + " for " + name);
        };
        return Tensor.of(values, info.shape());
    }

    private Object readValue(int type) throws IOException {
        return switch (type) {
            case 0 -> readUnsignedByte();
            case 1 -> file.readByte();
            case 2 -> readUnsignedShortLE();
            case 3 -> readShortLE();
            case 4 -> readIntLE();
            case 5 -> readIntLE();
            case 6 -> Float.intBitsToFloat(readIntLE());
            case 7 -> readBoolean();
            case 8 -> readString();
            case 9 -> readArray();
            case 10, 11 -> readLongLE();
            case 12 -> Double.longBitsToDouble(readLongLE());
            default -> throw new IOException("Unsupported GGUF metadata type: " + type);
        };
    }

    private ArrayList<Object> readArray() throws IOException {
        int elementType = readIntLE();
        long length = readLongLE();
        requireCount(length, MAX_ARRAY_ENTRIES, "array");
        ArrayList<Object> values = new ArrayList<>((int) length);
        for (long index = 0; index < length; index++) {
            values.add(readValue(elementType));
        }
        return values;
    }

    private String readString() throws IOException {
        long length = readLongLE();
        if (length < 0 || length > Integer.MAX_VALUE) {
            throw new IOException("GGUF string is too long");
        }
        byte[] bytes = new byte[(int) length];
        file.readFully(bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private float[] readF32(int size) throws IOException {
        float[] values = new float[size];
        for (int index = 0; index < size; index++) {
            values[index] = Float.intBitsToFloat(readIntLE());
        }
        return values;
    }

    private float[] readF16(int size) throws IOException {
        float[] values = new float[size];
        for (int index = 0; index < size; index++) {
            values[index] = halfToFloat(readUnsignedShortLE());
        }
        return values;
    }

    private float[] readQ4_0(int size) throws IOException {
        requireMultiple(size, 32, "Q4_0");
        float[] values = new float[size];
        for (int block = 0; block < size / 32; block++) {
            float scale = halfToFloat(readUnsignedShortLE());
            byte[] packed = new byte[16];
            file.readFully(packed);
            int base = block * 32;
            for (int index = 0; index < 16; index++) {
                int value = packed[index] & 0xff;
                values[base + index] = ((value & 0x0f) - 8) * scale;
                values[base + index + 16] = ((value >>> 4) - 8) * scale;
            }
        }
        return values;
    }

    private float[] readQ4_1(int size) throws IOException {
        requireMultiple(size, 32, "Q4_1");
        float[] values = new float[size];
        for (int block = 0; block < size / 32; block++) {
            float scale = halfToFloat(readUnsignedShortLE());
            float minimum = halfToFloat(readUnsignedShortLE());
            byte[] packed = new byte[16];
            file.readFully(packed);
            int base = block * 32;
            for (int index = 0; index < 16; index++) {
                int value = packed[index] & 0xff;
                values[base + index] = (value & 0x0f) * scale + minimum;
                values[base + index + 16] = (value >>> 4) * scale + minimum;
            }
        }
        return values;
    }

    private float[] readQ8_0(int size) throws IOException {
        requireMultiple(size, 32, "Q8_0");
        float[] values = new float[size];
        for (int block = 0; block < size / 32; block++) {
            float scale = halfToFloat(readUnsignedShortLE());
            int base = block * 32;
            for (int index = 0; index < 32; index++) {
                values[base + index] = file.readByte() * scale;
            }
        }
        return values;
    }

    private static int tensorSize(int[] shape) throws IOException {
        int size = 1;
        try {
            for (int dimension : shape) {
                size = Math.multiplyExact(size, dimension);
            }
        } catch (ArithmeticException exception) {
            throw new IOException("Tensor exceeds supported Java array size", exception);
        }
        return size;
    }

    private static void requireMultiple(int size, int blockSize, String type) throws IOException {
        if (size % blockSize != 0) {
            throw new IOException(type + " tensor size must be a multiple of " + blockSize);
        }
    }

    private int readUnsignedByte() throws IOException {
        return file.readUnsignedByte();
    }

    private int readUnsignedShortLE() throws IOException {
        int first = file.readUnsignedByte();
        int second = file.readUnsignedByte();
        return first | (second << 8);
    }

    private short readShortLE() throws IOException {
        return (short) readUnsignedShortLE();
    }

    private int readIntLE() throws IOException {
        return Integer.reverseBytes(file.readInt());
    }

    private long readLongLE() throws IOException {
        return Long.reverseBytes(file.readLong());
    }

    private boolean readBoolean() throws IOException {
        int value = file.readUnsignedByte();
        if (value > 1) {
            throw new IOException("Invalid GGUF boolean value: " + value);
        }
        return value == 1;
    }

    private static float halfToFloat(int half) {
        int sign = (half & 0x8000) << 16;
        int exponent = (half >>> 10) & 0x1f;
        int fraction = half & 0x03ff;
        int bits;
        if (exponent == 0) {
            if (fraction == 0) {
                bits = sign;
            } else {
                int adjustedExponent = -14;
                while ((fraction & 0x0400) == 0) {
                    fraction <<= 1;
                    adjustedExponent--;
                }
                fraction &= 0x03ff;
                bits = sign | ((adjustedExponent + 127) << 23) | (fraction << 13);
            }
        } else if (exponent == 0x1f) {
            bits = sign | 0x7f800000 | (fraction << 13);
        } else {
            bits = sign | ((exponent - 15 + 127) << 23) | (fraction << 13);
        }
        return Float.intBitsToFloat(bits);
    }

    private static void requireCount(long count, long maximum, String label) throws IOException {
        if (count < 0 || count > maximum) {
            throw new IOException("GGUF " + label + " count is outside supported range: " + count);
        }
    }

    @Override
    public void close() throws IOException {
        file.close();
    }

    /** Describes a tensor's logical shape, stored type, and data-section offset. */
    private record TensorInfo(String name, int[] shape, int type, long offset) {
    }
}
