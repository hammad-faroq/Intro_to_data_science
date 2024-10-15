def find_unique_users(transactions):
    """We are using sets in this function to keep track of unique users"""
    unique_users = set()
    for transaction in transactions:
        user_id = transaction[1]
        unique_users.add(user_id)
    return len(unique_users)
def highest_transaction(transactions):
    """This function will simply calc the highest transection amongst the other tracnsections"""
    highest = transactions[0]
    for transaction in transactions:
        if transaction[2] > highest[2]:
            highest = transaction
    return highest
def separate_ids(transactions):
    transaction_ids = []
    user_ids = []
    for transaction in transactions:
        transaction_ids.append(transaction[0])  # transaction_id_index= 0
        user_ids.append(transaction[1])         # user_id_index= 1
    return transaction_ids, user_ids  # Return two lists
def separate_ids_safe(transactions):
    """This function will check whether the tuple size is consistent or not -_-"""
    transaction_ids = []
    user_ids = []
    for transaction in transactions:
        if len(transaction) == 4:
            transaction_ids.append(transaction[0])
            user_ids.append(transaction[1])
        else:
            print(f"Skipping invalid transaction: {transaction}")
    return transaction_ids, user_ids
def main():
    """This is the main function of the program"""
    transactions = [
        (101, 1, 250.75, "2023-10-12"),
        (102, 2, 500.00, "2023-10-13"),
        (103, 1, 120.50, "2023-10-14"),
        (104, 3, 300.00, "2023-10-15")
    ]
    print(find_unique_users(transactions))
    print(highest_transaction(transactions))
    print(separate_ids(transactions))
main()