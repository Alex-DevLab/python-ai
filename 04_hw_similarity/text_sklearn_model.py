# Task N14

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class TextModel:
    def __init__(self, text: str):
        text_arr = text.split(".")
        self.cleaned_text = [res for s in text_arr if (res := s.strip())]
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.cleaned_text)

    def __vectorize_text(self, text: list[str]) -> np.ndarray:
        return self.vectorizer.transform(text).toarray()

    def get_answers(self, question: str, n_answers: int = 1) -> list[str]:
        answers_arr = []

        vectorized_text = self.__vectorize_text(self.cleaned_text)
        vectorized_question = self.__vectorize_text([question])

        similar_arr = cosine_similarity(vectorized_text, vectorized_question)
        result_idx = np.argsort(similar_arr, axis=0)[::-1]
        result_idx = result_idx.flatten()

        for val in result_idx:
            if similar_arr[val] == 0:
                break

            answers_arr.append(self.cleaned_text[val])

            if len(answers_arr) == n_answers:
                break

        return answers_arr


thinker = TextModel(
    "Learning a new skill can be both challenging and exciting. \
    Many people decide to learn something new because they want to improve their lives.\
    Some people learn a foreign language, while others learn programming, cooking, or photography.\
    At first, the new skill may seem difficult and confusing. \
    However, regular practice can make the process much easier.\
    It is important to set small and realistic goals when learning. \
    For example, a student can spend twenty minutes a day practicing instead of studying for several hours once a week. \
    Making mistakes is also a natural part of learning.\
    Mistakes show us what we need to improve.\
    It is useful to keep a record of progress because it helps us see how much we have learned. \
    Sometimes people lose motivation when they do not see quick results. \
    In this situation, taking a short break can help them return with more energy. \
    Learning becomes more enjoyable when people choose activities that are interesting to them.\
    With patience and regular practice, almost everyone can develop a useful new skill.\
    The most important thing is to continue learning even when progress seems slow."
)

# answers = thinker.get_answers("Why do many people decide to learn a new skill?", 5)
# answers = thinker.get_answers("What can make the learning process easier?", 2)
answers = thinker.get_answers("Why is it useful to set small and realistic goals?", 6)
# answers = thinker.get_answers("What can people do when they lose motivation?", 3)
# answers = thinker.get_answers("What is the most important thing when learning a new skill?", 5)
# answers = thinker.get_answers("What is the meaning of life?", 5)
print(answers)
