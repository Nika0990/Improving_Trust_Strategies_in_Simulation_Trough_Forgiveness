import numpy as np
import json

################################
# CONSTS
################################

REVIEWS = 0
BOT_ACTION = 1
USER_DECISION = 2
REVIEW_ID = 3


################################


def forgiving_trustful_with_max_threshold(history_window, quality_threshold, forgiveness_count_max):
    def func(information):
        forgiveness_counter = 0
        for previous_round in information["previous_rounds"][-history_window:]:
            if not ((previous_round[BOT_ACTION] >= 8 and previous_round[REVIEWS].mean() >= 8) or
                    (previous_round[BOT_ACTION] <= 8 and previous_round[REVIEWS].mean() < 8)):
                forgiveness_counter += 1
            if forgiveness_counter > forgiveness_count_max:
                return 0

        if information["bot_message"] >= quality_threshold:
            return 1
        else:
            return 0

    return func


def forgiving_trustful_with_llm(history_window, quality_threshold, forgiveness_threshold):
    def func(information):
        with open(f"data/baseline_proba2go.txt", 'r') as file:
            proba2go = json.load(file)
            proba2go = {int(k): v for k, v in proba2go.items()}
        review_llm_score = proba2go[information["review_id"]]

        def check_condition(r):
            llm_score = proba2go[r[REVIEW_ID]]
            review_mean = r[REVIEWS].mean()
            if (llm_score >= 0.8 and review_mean >= 8) or (llm_score <= 0.8 and review_mean < 8):
                return 1
            elif abs(llm_score * 10 - review_mean) < forgiveness_threshold:
                return 1
            return 0

        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(
            np.array([check_condition(r) for r in information["previous_rounds"][-history_window:]])) == 1:
            if review_llm_score * 10 >= quality_threshold:
                return 1
            else:
                return 0
        else:
            return 0

    return func



def correct_action(information):
    if information["hotel_value"] >= 8:
        return 1
    else:
        return 0


def random_action(information):
    return np.random.randint(2)


def user_rational_action(information):
    if information["bot_message"] >= 8:
        return 1
    else:
        return 0


def user_picky(information):
    if information["bot_message"] >= 9:
        return 1
    else:
        return 0


def user_sloppy(information):
    if information["bot_message"] >= 7:
        return 1
    else:
        return 0


def user_short_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or (information["previous_rounds"][-1][BOT_ACTION] >= 8 and
                information["previous_rounds"][-1][REVIEWS].mean() >= 8) \
            or (information["previous_rounds"][-1][BOT_ACTION] < 8 and
                information["previous_rounds"][-1][REVIEWS].mean() < 8):  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def user_picky_short_t4t(information):
    if information["bot_message"] >= 9 or ((information["bot_message"] >= 8) and (
            len(information["previous_rounds"]) == 0 or (
            information["previous_rounds"][-1][REVIEWS].mean() >= 8))):  # good hotel
        return 1
    else:
        return 0


def user_hard_t4t(information):
    if len(information["previous_rounds"]) == 0 \
            or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                 or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                information["previous_rounds"]])) == 1:  # cooperation
        if information["bot_message"] >= 8:  # good hotel
            return 1
        else:
            return 0
    else:
        return 0


def history_and_review_quality(history_window, quality_threshold):
    def func(information):
        if len(information["previous_rounds"]) == 0 \
                or history_window == 0 \
                or np.min(np.array([((r[BOT_ACTION] >= 8 and r[REVIEWS].mean() >= 8)
                                     or (r[BOT_ACTION] <= 8 and r[REVIEWS].mean() < 8)) for r in
                                    information["previous_rounds"][
                                    -history_window:]])) == 1:  # cooperation from *result's* perspective
            if information["bot_message"] >= quality_threshold:  # good hotel from user's perspective
                return 1
            else:
                return 0
        else:
            return 0

    return func


def topic_based(positive_topics, negative_topics, quality_threshold):
    def func(information):
        review_personal_score = information["bot_message"]
        for rank, topic in enumerate(positive_topics):
            review_personal_score += int(information["review_features"].loc[topic])*2/(rank+1)
        for rank, topic in enumerate(negative_topics):
            review_personal_score -= int(information["review_features"].loc[topic])*2/(rank+1)
        if review_personal_score >= quality_threshold:  # good hotel from user's perspective
            return 1
        else:
            return 0
    return func


def LLM_based(is_stochastic):
    with open(f"data/baseline_proba2go.txt", 'r') as file:
        proba2go = json.load(file)
        proba2go = {int(k): v for k, v in proba2go.items()}

    if is_stochastic:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(np.random.rand() <= review_llm_score)
        return func
    else:
        def func(information):
            review_llm_score = proba2go[information["review_id"]]
            return int(review_llm_score >= 0.5)
        return func