package com.javallm.inference;

import com.javallm.model.LoadedModel;
import com.javallm.tensor.Tensor;
import com.javallm.transformer.KeyValueCache;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.random.RandomGenerator;

/**
 * Autoregressive CPU token generation with temperature, top-k, top-p, and KV caching.
 *
 * <p>The prompt is tokenized and evaluated once to prefill the cache. Each next
 * token is sampled from the final-position logits and, unless generation is
 * complete, evaluated incrementally to update the cache. Generation stops at
 * the configured token limit or the tokenizer's end-of-sequence token.</p>
 */
public final class TokenGenerator implements TextGenerator {

    private static final Logger LOGGER = Logger.getLogger(TokenGenerator.class.getName());

    private final LoadedModel loadedModel;
    private final RandomGenerator random;

    public TokenGenerator(LoadedModel loadedModel) {
        this(loadedModel, RandomGenerator.getDefault());
    }

    public TokenGenerator(LoadedModel loadedModel, RandomGenerator random) {
        this.loadedModel = Objects.requireNonNull(loadedModel, "Loaded model cannot be null");
        this.random = Objects.requireNonNull(random, "Random generator cannot be null");
    }

    @Override
    public GenerationResult generate(String prompt, GenerationConfig config) {
        Objects.requireNonNull(prompt, "Prompt cannot be null");
        Objects.requireNonNull(config, "Generation config cannot be null");

        long generationStart = System.nanoTime();
        LOGGER.info("TOKENIZATION: prompt: " + prompt);
        List<Integer> inputTokens = loadedModel.tokenizer().encode(prompt);
        if (inputTokens.isEmpty()) {
            LOGGER.warning("Prompt tokenization produced no tokens.");
            throw new IllegalArgumentException("Prompt must encode to at least one token");
        }
        if (inputTokens.size() > loadedModel.contextLength()) {
            LOGGER.log(Level.WARNING, "Prompt has {0} tokens, exceeding model context length {1}.",
                    new Object[]{inputTokens.size(), loadedModel.contextLength()});
            throw new IllegalArgumentException("Prompt exceeds model context length");
        }
        LOGGER.log(Level.INFO, "TOKENIZATION: token count={0}, token IDs={1} (context capacity {2}).",
                new Object[]{inputTokens.size(), inputTokens, loadedModel.contextLength()});

        int[] promptIds = new int[inputTokens.size()];
        for (int index = 0; index < inputTokens.size(); index++) {
            promptIds[index] = inputTokens.get(index);
        }

        KeyValueCache cache = loadedModel.model().newKeyValueCache(loadedModel.contextLength());
        LOGGER.info("PREFILL: input token IDs to forwardWithCache(): " + Arrays.toString(promptIds));
        LOGGER.info("PREFILL: starting decoder and key/value cache prefill.");
        long prefillStart = System.nanoTime();
        Tensor logits = loadedModel.model().forwardWithCache(promptIds, cache);
        LOGGER.info("PREFILL: logits tensor shape=" + Arrays.toString(logits.shape()));
        for (int row = 0; row < logits.dimension(0); row++) {
            LOGGER.info("PREFILL: logits row " + row + " first values=" + tensorRowPrefix(logits, row, 10));
        }
        LOGGER.log(Level.INFO, "PREFILL: cached {0} tokens in {1} ms.",
                new Object[]{cache.sequenceLength(), elapsedMilliseconds(prefillStart)});
        LOGGER.log(Level.INFO, "Sampling settings: maxNewTokens={0}, temperature={1}, topK={2}, topP={3}.",
                new Object[]{config.maximumNewTokens(), config.temperature(), config.topK(), config.topP()});
        ArrayList<Integer> generated = new ArrayList<>(config.maximumNewTokens());
        String stopReason = "maximum new-token count reached";

        for (int generatedIndex = 0; generatedIndex < config.maximumNewTokens(); generatedIndex++) {
            String phase = generatedIndex == 0 ? "PREFILL" : "GENERATION";
            float[] samplingLogits = lastLogits(logits);
            LOGGER.info(phase + ": lastLogits array shape=[" + samplingLogits.length + "], first values="
                    + arrayPrefix(samplingLogits, 20));
            int nextToken = sample(samplingLogits, config, phase);
            int eosTokenId = loadedModel.tokenizer().eosTokenId();
            if (eosTokenId >= 0 && nextToken == eosTokenId) {
                stopReason = "end-of-sequence token sampled";
                LOGGER.info("Sampled the end-of-sequence token; stopping generation.");
                break;
            }
            generated.add(nextToken);
            LOGGER.log(Level.INFO, phase + ": generated token {0}/{1}: id={2}, text=\"{3}\".",
                    new Object[]{generated.size(), config.maximumNewTokens(), nextToken, decodedToken(nextToken)});

            if (generatedIndex + 1 < config.maximumNewTokens()) {
                if (cache.sequenceLength() >= loadedModel.contextLength()) {
                    stopReason = "model context length reached";
                    LOGGER.info("Model context length reached; stopping generation.");
                    break;
                }
                long decodeStepStart = System.nanoTime();
                int[] generationInputIds = new int[]{nextToken};
                LOGGER.info("GENERATION: input token ID to forwardWithCache(): " + Arrays.toString(generationInputIds)
                        + ", KV cache sequence length before=" + cache.sequenceLength());
                logits = loadedModel.model().forwardWithCache(generationInputIds, cache);
                LOGGER.info("GENERATION: logits tensor shape=" + Arrays.toString(logits.shape())
                        + ", KV cache sequence length after=" + cache.sequenceLength());
                LOGGER.log(Level.FINE, "GENERATION: incremental forward pass took {0} ms.",
                        elapsedMilliseconds(decodeStepStart));
            }
        }

        long decodeStart = System.nanoTime();
        String decodedText = loadedModel.tokenizer().decode(generated);
        LOGGER.log(Level.INFO, "Decoded {0} generated tokens to {1} characters.",
                new Object[]{generated.size(), decodedText.length()});
        LOGGER.log(Level.INFO, "Generation complete: {0}; elapsed {1} ms.",
                new Object[]{stopReason, elapsedMilliseconds(generationStart)});
        LOGGER.log(Level.FINE, "Generated-token decoding took {0} ms.", elapsedMilliseconds(decodeStart));
        return new GenerationResult(decodedText, generated);
    }

    private int sample(float[] logits, GenerationConfig config, String phase) {
        if (logits.length == 0) {
            throw new IllegalArgumentException("Cannot sample empty logits");
        }
        ArrayList<Candidate> candidates = new ArrayList<>(logits.length);
        for (int tokenId = 0; tokenId < logits.length; tokenId++) {
            double logit = Float.isNaN(logits[tokenId]) ? Double.NEGATIVE_INFINITY : logits[tokenId];
            candidates.add(new Candidate(tokenId, logit));
        }
        candidates.sort(Comparator.comparingDouble(Candidate::logit).reversed());
        logTopCandidates(phase, "top candidates before filtering", candidates, null, Math.min(10, candidates.size()));

        if (config.temperature() == 0.0) {
            Candidate selected = candidates.get(0);
            LOGGER.info(phase + ": top-K and top-P filtering bypassed because temperature is zero.");
            LOGGER.info(phase + ": selected token id=" + selected.tokenId() + ", text=\""
                    + decodedToken(selected.tokenId()) + "\".");
            return selected.tokenId();
        }
        if (config.topK() > 0 && candidates.size() > config.topK()) {
            candidates.subList(config.topK(), candidates.size()).clear();
        }
        LOGGER.info(phase + ": candidates surviving top-K (" + config.topK() + "): " + candidateIds(candidates));

        int positiveInfinityCount = 0;
        for (Candidate candidate : candidates) {
            if (candidate.logit() == Double.POSITIVE_INFINITY) {
                positiveInfinityCount++;
            }
        }
        if (positiveInfinityCount > 0) {
            candidates.removeIf(candidate -> candidate.logit() != Double.POSITIVE_INFINITY);
            double[] probabilities = new double[candidates.size()];
            Arrays.fill(probabilities, 1.0 / candidates.size());
            logTopCandidates(phase, "probabilities after temperature/softmax", candidates, probabilities,
                    Math.min(10, candidates.size()));
            LOGGER.info(phase + ": candidates surviving top-P: " + candidateIds(candidates)
                    + " (positive-infinity handling bypasses top-P truncation).");
            int selectedToken = candidates.get(random.nextInt(candidates.size())).tokenId();
            logSelectedToken(phase, selectedToken);
            return selectedToken;
        }

        double maximum = candidates.get(0).logit();
        double[] weights = new double[candidates.size()];
        double totalWeight = 0.0;
        if (maximum == Double.NEGATIVE_INFINITY) {
            java.util.Arrays.fill(weights, 1.0);
            totalWeight = weights.length;
        } else {
            for (int index = 0; index < candidates.size(); index++) {
                double weight = Math.exp((candidates.get(index).logit() - maximum) / config.temperature());
                weights[index] = weight;
                totalWeight += weight;
            }
        }

        double[] probabilities = new double[weights.length];
        for (int index = 0; index < weights.length; index++) {
            probabilities[index] = weights[index] / totalWeight;
        }
        logTopCandidates(phase, "probabilities after temperature/softmax", candidates, probabilities,
                Math.min(10, candidates.size()));

        int retainedCount = weights.length;
        if (config.topP() < 1.0) {
            double cumulativeProbability = 0.0;
            retainedCount = 0;
            for (int index = 0; index < weights.length; index++) {
                cumulativeProbability += weights[index] / totalWeight;
                retainedCount++;
                if (cumulativeProbability >= config.topP()) {
                    break;
                }
            }
        }
        LOGGER.info(phase + ": candidates surviving top-P (" + config.topP() + "): "
                + candidateIds(candidates.subList(0, retainedCount)));

        double retainedWeight = 0.0;
        for (int index = 0; index < retainedCount; index++) {
            retainedWeight += weights[index];
        }
        double selection = random.nextDouble() * retainedWeight;
        for (int index = 0; index < retainedCount; index++) {
            selection -= weights[index];
            if (selection <= 0.0) {
                int selectedToken = candidates.get(index).tokenId();
                logSelectedToken(phase, selectedToken);
                return selectedToken;
            }
        }
        int selectedToken = candidates.get(retainedCount - 1).tokenId();
        logSelectedToken(phase, selectedToken);
        return selectedToken;
    }

    private void logTopCandidates(String phase, String label, List<Candidate> candidates,
                                  double[] probabilities, int count) {
        for (int index = 0; index < count; index++) {
            Candidate candidate = candidates.get(index);
            String probability = probabilities == null ? "not computed" : Double.toString(probabilities[index]);
            LOGGER.info(phase + ": " + label + " [rank=" + (index + 1) + "]: tokenId=" + candidate.tokenId()
                    + ", text=\"" + decodedToken(candidate.tokenId()) + "\", logit=" + candidate.logit()
                    + ", probability=" + probability);
        }
    }

    private String candidateIds(List<Candidate> candidates) {
        ArrayList<Integer> tokenIds = new ArrayList<>(candidates.size());
        for (Candidate candidate : candidates) {
            tokenIds.add(candidate.tokenId());
        }
        return tokenIds.toString();
    }

    private void logSelectedToken(String phase, int tokenId) {
        LOGGER.info(phase + ": selected token id=" + tokenId + ", text=\"" + decodedToken(tokenId) + "\".");
    }

    private String decodedToken(int tokenId) {
        return loadedModel.tokenizer().decode(List.of(tokenId));
    }

    private static String tensorRowPrefix(Tensor tensor, int row, int count) {
        int length = Math.min(count, tensor.dimension(1));
        float[] values = new float[length];
        for (int column = 0; column < length; column++) {
            values[column] = tensor.get(row, column);
        }
        return Arrays.toString(values);
    }

    private static String arrayPrefix(float[] values, int count) {
        return Arrays.toString(Arrays.copyOf(values, Math.min(count, values.length)));
    }

    private static float[] lastLogits(Tensor logits) {
        if (logits.rank() != 2 || logits.dimension(0) <= 0) {
            throw new IllegalArgumentException("Decoder logits must have shape [sequence, vocabulary]");
        }
        int vocabularySize = logits.dimension(1);
        float[] result = new float[vocabularySize];
        int lastPosition = logits.dimension(0) - 1;
        for (int tokenId = 0; tokenId < vocabularySize; tokenId++) {
            result[tokenId] = logits.get(lastPosition, tokenId);
        }
        return result;
    }

    private static long elapsedMilliseconds(long startTime) {
        return (System.nanoTime() - startTime) / 1_000_000L;
    }

    /** Associates a vocabulary entry with its sampling score. */
    private record Candidate(int tokenId, double logit) {
    }
}
