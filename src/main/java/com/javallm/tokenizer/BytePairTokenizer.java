package com.javallm.tokenizer;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * GPT-2-style byte-level BPE tokenizer backed by vocabulary and merge arrays
 * stored in GGUF metadata.
 *
 * <p>Encoding applies the model's pre-tokenizer, maps UTF-8 bytes to the GPT-2
 * byte alphabet, and repeatedly applies ranked BPE merges. Decoding reverses the
 * byte mapping and joins the resulting UTF-8 text.</p>
 */
public final class BytePairTokenizer implements Tokenizer {

    private static final String METADATA_MODEL_KEY = "tokenizer.ggml.model";
    private static final String METADATA_PRETOKENIZER_KEY = "tokenizer.ggml.pre";
    private static final String METADATA_TOKENS_KEY = "tokenizer.ggml.tokens";
    private static final String METADATA_MERGES_KEY = "tokenizer.ggml.merges";
    private static final String METADATA_BOS_TOKEN_ID_KEY = "tokenizer.ggml.bos_token_id";
    private static final String METADATA_EOS_TOKEN_ID_KEY = "tokenizer.ggml.eos_token_id";
    private static final String METADATA_UNKNOWN_TOKEN_ID_KEY = "tokenizer.ggml.unknown_token_id";
    private static final String METADATA_ADD_BOS_TOKEN_KEY = "tokenizer.ggml.add_bos_token";

    private static final Pattern PRE_TOKEN_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\\p{L}+| ?\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );
    private static final Pattern SMOLLM_PRE_TOKEN_PATTERN = Pattern.compile(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );

    private final List<String> tokens;
    private final Map<String, Integer> tokenIds;
    private final Map<String, Integer> mergeRanks;
    private final List<String> specialTokens;
    private final int[] byteToCodePoint;
    private final int[] codePointToByte;
    private final Pattern preTokenPattern;
    private final int bosTokenId;
    private final int eosTokenId;
    private final int unknownTokenId;
    private final boolean addBosToken;

    public BytePairTokenizer(Map<String, Object> metadata) {
        Objects.requireNonNull(metadata, "Metadata cannot be null");
        Object model = metadata.get(METADATA_MODEL_KEY);
        if (!(model instanceof String modelName) || !modelName.equals("gpt2")) {
            throw new IllegalArgumentException("Only GPT-2 byte-level BPE tokenizers are supported");
        }
        Object preTokenizerValue = metadata.get(METADATA_PRETOKENIZER_KEY);
        String preTokenizer = preTokenizerValue instanceof String string ? string : "gpt2";
        if (preTokenizer.equals("smollm")) {
            this.preTokenPattern = SMOLLM_PRE_TOKEN_PATTERN;
        } else if (preTokenizer.equals("gpt2")) {
            this.preTokenPattern = PRE_TOKEN_PATTERN;
        } else {
            throw new IllegalArgumentException("Unsupported GPT-2 byte-level pre-tokenizer: " + preTokenizer);
        }

        this.tokens = stringList(metadata.get(METADATA_TOKENS_KEY), METADATA_TOKENS_KEY);
        List<String> merges = stringList(metadata.get(METADATA_MERGES_KEY), METADATA_MERGES_KEY);
        this.tokenIds = new HashMap<>();
        this.specialTokens = new ArrayList<>();
        for (int tokenId = 0; tokenId < tokens.size(); tokenId++) {
            String token = tokens.get(tokenId);
            tokenIds.putIfAbsent(token, tokenId);
            if (isSpecialToken(token)) {
                specialTokens.add(token);
            }
        }
        this.mergeRanks = new HashMap<>();
        for (int rank = 0; rank < merges.size(); rank++) {
            mergeRanks.putIfAbsent(merges.get(rank), rank);
        }

        this.byteToCodePoint = new int[256];
        this.codePointToByte = new int[512];
        java.util.Arrays.fill(codePointToByte, -1);
        int extraCodePoint = 256;
        for (int value = 0; value < 256; value++) {
            boolean directlyRepresented = (value >= 33 && value <= 126)
                    || (value >= 161 && value <= 172)
                    || (value >= 174 && value <= 255);
            int codePoint = directlyRepresented ? value : extraCodePoint++;
            byteToCodePoint[value] = codePoint;
            codePointToByte[codePoint] = value;
        }

        this.bosTokenId = intValue(metadata, METADATA_BOS_TOKEN_ID_KEY, -1);
        this.eosTokenId = intValue(metadata, METADATA_EOS_TOKEN_ID_KEY, -1);
        this.unknownTokenId = intValue(metadata, METADATA_UNKNOWN_TOKEN_ID_KEY, -1);
        this.addBosToken = booleanValue(metadata, METADATA_ADD_BOS_TOKEN_KEY, false);
    }

    @Override
    public int vocabularySize() {
        return tokens.size();
    }

    @Override
    public int bosTokenId() {
        return bosTokenId;
    }

    @Override
    public int eosTokenId() {
        return eosTokenId;
    }

    @Override
    public boolean addsBosToken() {
        return addBosToken;
    }

    @Override
    public List<Integer> encode(String text) {
        Objects.requireNonNull(text, "Text cannot be null");
        ArrayList<Integer> ids = new ArrayList<>();
        int cursor = 0;
        while (cursor < text.length()) {
            SpecialMatch special = findNextSpecial(text, cursor);
            int normalEnd = special == null ? text.length() : special.start();
            if (normalEnd > cursor) {
                encodeNormalText(text.substring(cursor, normalEnd), ids);
            }
            if (special != null) {
                Integer specialId = tokenIds.get(special.token());
                if (specialId == null) {
                    throw new IllegalStateException("Special token is missing from the vocabulary: " + special.token());
                }
                ids.add(specialId);
                cursor = special.end();
            } else {
                cursor = text.length();
            }
        }
        if (text.isEmpty()) {
            return ids;
        }
        if (addBosToken && bosTokenId >= 0) {
            ids.add(0, bosTokenId);
        }
        return ids;
    }

    @Override
    public String decode(List<Integer> tokenSequence) {
        Objects.requireNonNull(tokenSequence, "Token sequence cannot be null");
        StringBuilder result = new StringBuilder();
        ByteArrayOutputStream pendingBytes = new ByteArrayOutputStream();
        for (Integer tokenIdValue : tokenSequence) {
            if (tokenIdValue == null || tokenIdValue < 0 || tokenIdValue >= tokens.size()) {
                throw new IllegalArgumentException("Token ID is outside the vocabulary: " + tokenIdValue);
            }
            String token = tokens.get(tokenIdValue);
            if (isSpecialToken(token)) {
                flushBytes(pendingBytes, result);
                result.append(token);
            } else {
                appendTokenBytes(token, pendingBytes);
            }
        }
        flushBytes(pendingBytes, result);
        return result.toString();
    }

    @Override
    public String token(int tokenId) {
        if (tokenId < 0 || tokenId >= tokens.size()) {
            throw new IllegalArgumentException("Token ID is outside the vocabulary: " + tokenId);
        }
        return tokens.get(tokenId);
    }

    private void encodeNormalText(String text, List<Integer> output) {
        Matcher matcher = preTokenPattern.matcher(text);
        while (matcher.find()) {
            String piece = matcher.group();
            ArrayList<String> bytePieces = toBytePieces(piece);
            applyBpe(bytePieces);
            for (String bytePiece : bytePieces) {
                Integer tokenId = tokenIds.get(bytePiece);
                if (tokenId == null) {
                    if (unknownTokenId < 0) {
                        throw new IllegalArgumentException("Tokenizer vocabulary cannot encode token piece: " + bytePiece);
                    }
                    output.add(unknownTokenId);
                } else {
                    output.add(tokenId);
                }
            }
        }
    }

    private ArrayList<String> toBytePieces(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        ArrayList<String> pieces = new ArrayList<>(bytes.length);
        for (byte value : bytes) {
            int unsignedValue = value & 0xff;
            pieces.add(new String(Character.toChars(byteToCodePoint[unsignedValue])));
        }
        return pieces;
    }

    private void applyBpe(ArrayList<String> pieces) {
        while (pieces.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            String bestPair = null;
            for (int index = 0; index < pieces.size() - 1; index++) {
                String pair = pieces.get(index) + " " + pieces.get(index + 1);
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }
            if (bestPair == null) {
                break;
            }
            int separator = bestPair.indexOf(' ');
            String first = bestPair.substring(0, separator);
            String second = bestPair.substring(separator + 1);
            for (int index = 0; index < pieces.size() - 1;) {
                if (pieces.get(index).equals(first) && pieces.get(index + 1).equals(second)) {
                    pieces.set(index, first + second);
                    pieces.remove(index + 1);
                } else {
                    index++;
                }
            }
        }
    }

    private SpecialMatch findNextSpecial(String text, int fromIndex) {
        SpecialMatch best = null;
        for (String token : specialTokens) {
            int start = text.indexOf(token, fromIndex);
            if (start >= 0 && (best == null || start < best.start() || (start == best.start() && token.length() > best.token().length()))) {
                best = new SpecialMatch(start, start + token.length(), token);
            }
        }
        return best;
    }

    private void appendTokenBytes(String token, ByteArrayOutputStream output) {
        for (int index = 0; index < token.length();) {
            int codePoint = token.codePointAt(index);
            if (codePoint >= codePointToByte.length || codePointToByte[codePoint] < 0) {
                throw new IllegalArgumentException("Token contains a character outside GPT-2 byte encoding");
            }
            output.write(codePointToByte[codePoint]);
            index += Character.charCount(codePoint);
        }
    }

    private static void flushBytes(ByteArrayOutputStream bytes, StringBuilder output) {
        output.append(bytes.toString(StandardCharsets.UTF_8));
        bytes.reset();
    }

    private static boolean isSpecialToken(String token) {
        return (token.startsWith("<|") && token.endsWith("|>"))
                || token.equals("<s>")
                || token.equals("</s>");
    }

    private static List<String> stringList(Object value, String key) {
        if (!(value instanceof List<?> values)) {
            throw new IllegalArgumentException("GGUF tokenizer metadata is missing array " + key);
        }
        ArrayList<String> strings = new ArrayList<>(values.size());
        for (Object item : values) {
            if (!(item instanceof String string)) {
                throw new IllegalArgumentException("GGUF tokenizer array contains a non-string value: " + key);
            }
            strings.add(string);
        }
        return List.copyOf(strings);
    }

    private static int intValue(Map<String, Object> metadata, String key, int fallback) {
        Object value = metadata.get(key);
        if (value instanceof Number number) {
            long longValue = number.longValue();
            if (longValue >= Integer.MIN_VALUE && longValue <= Integer.MAX_VALUE) {
                return (int) longValue;
            }
        }
        return fallback;
    }

    private static boolean booleanValue(Map<String, Object> metadata, String key, boolean fallback) {
        Object value = metadata.get(key);
        return value instanceof Boolean booleanValue ? booleanValue : fallback;
    }

    /** Identifies a recognized special-token span in prompt text. */
    private record SpecialMatch(int start, int end, String token) {
    }
}
