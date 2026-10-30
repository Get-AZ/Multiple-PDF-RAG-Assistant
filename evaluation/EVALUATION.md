# Evaluation plan and evidence record

This release preserves the supplied source. Historical multi-PDF outputs showed successful answers and missing-evidence failures. No exact-source aggregate score is available. Single-PDF test scores must not be reused here.

## Acceptance procedure
Freeze source hash, dependency versions, resolved model IDs, PDF hashes, chunk parameters and prompt. Run a baseline without tuning. Record question, expected facts, source pages, retrieved passages, answer, latency and API errors. Score each requested fact, citation correctness and unsupported claims. Keep revised questions and tuned retests separate.

## Suggested cases
1. YOLO: What FPS are reported for YOLO and Fast YOLO? Expected: 45 and 155 (table).
2. YOLO: What are the convolutional layer counts? Expected: 24 and 9.
3. Transformer: What are the encoder sublayers? Expected: multi-head self-attention and feed-forward network.
4. Transformer: Why use positional encodings? Expected: inject sequence-position information.
5. Compare the task, input and output of YOLO and the Transformer. Require evidence from both files.
6. Compare image-position prediction and token positional encoding. Require evidence from both files.
7. Compare the roles of convolution and self-attention. Require evidence from both files.
8. Compare YOLO speed, accuracy and depth. Require evidence from multiple sections.
9. Explain decoder masking and the sources of queries/keys/values. Require multiple passages.
10. What exact electricity dollar cost trained the base Transformer? Expected: absent.
11. What FPS did original YOLO obtain on an RTX 4090? Expected: absent.
12. What English-to-Amharic BLEU does the Transformer paper report? Expected: absent.

Verify page positions against the actual PDF editions before scoring. No results are prefilled. Separately test changing the upload selection, failed reprocessing, empty text, cleared chat and one-chunk collections.
