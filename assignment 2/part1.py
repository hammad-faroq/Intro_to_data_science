def filter_users(user_list):
    filtered_names = []
    for user in user_list:
        user_id, user_name, age, country = user#multi variable assignment inn one go
        if age > 30 and country in ['USA', 'Canada']:
            filtered_names.append(user_name)
    return filtered_names

def top_10_oldest_users(user_list):
    sorted_list = sorted(user_list, key=lambda x: x[2], reverse=True)#this will sort it by the age :)
    return sorted_list[:10]  # Returning the top 10

def find_duplicate_names(user_list):
    name_count = {}
    duplicates = []
    for user in user_list:
        user_name = user[1]
        if user_name in name_count:
            name_count[user_name] += 1
        else:
            name_count[user_name] = 1
    for name, count in name_count.items():
        if count > 1:
            duplicates.append(name)
    return duplicates
def main():
    user_data = [
        (101, "Alice", 25, "USA"),
        (102, "Bob", 32, "Canada"),
        (103, "Charlie", 28, "UK"),
        (104, "David", 35, "USA"),
        (105, "Eve", 40, "Canada"),
        (106, "Frank", 29, "USA"),
        (107, "Grace", 33, "Canada"),
        (108, "Heidi", 31, "Germany"),
        (109, "Ivan", 37, "USA"),
        (110, "Judy", 30, "Canada")
    ]
    filtered_names = filter_users(user_data)
    print("Filtered users' names (older than 30 from USA/Canada):", filtered_names)
    oldest_users = top_10_oldest_users(user_data)
    print("Top 10 oldest users:", oldest_users)
    duplicate_names = find_duplicate_names(user_data)
    print("Duplicate names found:", duplicate_names)
main()