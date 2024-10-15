def users_in_both_pages(page_A, page_B):
    return page_A & page_B
def users_in_either_but_not_both(page_A, page_C):
    return page_A ^ page_C
def update_page_A(page_A, new_user_ids):
    page_A.update(new_user_ids)
def remove_users_from_page_B(page_B, user_ids_to_remove):
    page_B.difference_update(user_ids_to_remove)
def main():
    page_A = {1, 2, 3, 4}
    page_B = {3, 4, 5}
    page_C = {2, 4, 6}
    print(users_in_both_pages(page_A, page_B))
    print(users_in_either_but_not_both(page_A, page_C))
    update_page_A(page_A, {7, 8})
    print(page_A)
    remove_users_from_page_B(page_B, {4, 5})
    print(page_B)
main()