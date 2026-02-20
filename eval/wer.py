from typing import Tuple

import jiwer


# WER (Word Error Rate):
# - Substitutions: is the total of words replaced in the hypothesis transcription.
# For example, if the reference transcription is “cat” and the recognized word is “bat”, then there is one substitution error.
# - Deletions: is the total of words missing in the hypothesis transcription.
# https://medium.com/@johnidouglasmarangon/how-to-calculate-the-word-error-rate-in-python-ce0751a46052


def wer_calc(ref, hypo):
    ref_words = ref.split()
    hyp_words = hypo.split()
    n, m = len(ref_words), len(hyp_words)

    # Initialize DP matrix for edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # Deletions
    for j in range(m + 1):
        dp[0][j] = j  # Insertions
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1  # Substitution cost
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost  # Substitution or match
            )
    edits = dp[n][m]  # Total S + D + I
    wer = edits / n if n > 0 else 0  # Final WER calculation.
    return wer


def test_wer_calc():
    ref = "I am 32 years old and I am a software developer"  # Ground truth (transcription)
    hypo = "I am 32 year old and I am a big software developer"  # Predicted transcription (model output)

    wer = wer_calc(ref, hypo)
    print(f"WER: {wer:.2%}")  # WER


Normalizer = jiwer.Compose(
        [
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )


def jiwer_wer(ref: str, hypo: str, normalize: bool = True) -> float:

    wer = jiwer.wer(
        ref,
        hypo,
        reference_transform=Normalizer if normalize else None,
        hypothesis_transform=Normalizer if normalize else None,
    )

    return wer


def test_jiwer_wer():
    ref = "I am 32 years old and I am a software developer"
    hypo = "I am 32 year old and I am a big software developer"

    wer = jiwer_wer(ref, hypo)
    print(f"JIWER WER: {wer:.2%}")

    homebrewed_wer = wer_calc(ref, hypo)
    print(f"Homebrewed WER: {homebrewed_wer:.2%}")


if __name__ == "__main__":
    test_jiwer_wer()
