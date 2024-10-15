def filter_high_ratings(feedback):
    high_ratings = {}
    for user_id, details in feedback.items():
        if details['rating'] >= 4:
            high_ratings[user_id] = details['rating']
    return high_ratings
def top_5_users(feedback):
    sorted_feedback = sorted(feedback.items(), key=lambda x: x[1]['rating'], reverse=True)
    top_5 = sorted_feedback[:5]
    return top_5  # Return the top 5 sorted feedback
def combine_feedback(*feedback_dicts):
    combined_feedback = {}
    for feedback in feedback_dicts:
        for user_id, details in feedback.items():
            if user_id in combined_feedback:
                combined_feedback[user_id]['rating'] = max(combined_feedback[user_id]['rating'], details['rating'])
                combined_feedback[user_id]['comments'] += " " + details['comments']
            else:
                combined_feedback[user_id] = details.copy()
    return combined_feedback
def users_with_rating_above_3(feedback):
    return {user_id: details['rating'] for user_id, details in feedback.items() if details['rating'] > 3}
def main():
    feedback1 = {
        101: {'rating': 5, 'comments': "Great service!"},
        102: {'rating': 3, 'comments': "It was okay."},
        103: {'rating': 4, 'comments': "Good experience."},
    }
    feedback2 = {
        101: {'rating': 4, 'comments': "Updated feedback."},
        104: {'rating': 5, 'comments': "Amazing support."},
        105: {'rating': 2, 'comments': "Not satisfied."},
    }
    print(filter_high_ratings(feedback1))
    print(top_5_users(feedback1))
    print(combine_feedback(feedback1, feedback2))
    print(users_with_rating_above_3(feedback1))
main()