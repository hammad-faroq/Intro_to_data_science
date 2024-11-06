import numpy as np
data_type = [('name', 'U10'), ('score', 'i4'), ('level', 'i4')]
players = np.array([
    ('Alice', 1500, 10),
    ('Bob', 1200,8),
    ('Charlie', 1800, 12)
], dtype=data_type)

print("Initial player data:\n", players)
# 2.Updating highest level score
max_level_index = np.argmax(players['level'])
players['score'][max_level_index] += 121  #increment for highest
print(" Updated the highest level player's score:\n", players)
# 3. Append a new player data
new_player = np.array([('David', 1700, 9)], dtype=data_type)
players = np.append(players, new_player)
print("\nAfter appending new player:\n", players)
# 4.Inssert at index 1
new_player2 = np.array([('Eve', 1600, 7)], dtype=data_type)
players = np.insert(players, 1, new_player2)
print("\nAfter inserting new player at index 1:\n", players)
# 5del minimum from the data set
min_score_index = np.argmin(players['score'])
players = np.delete(players, min_score_index)
print("\nAfter deleting the player with the lowest score:\n", players)
"""_______________________________________________________________"""
#2 arrays of numpy created!
visitors = np.array([120, 80, 95, 130, 60, 85, 100])
pages_viewed = np.array([300, 150, 210, 400, 90, 140, 250])
print("Visitors array:\n", visitors)
print("Pages viewed array:\n", pages_viewed)
# 2. Shalow copy and eep copy
visitors_shallow_copy = visitors.view()  # Shallow copy
pages_viewed_deep_copy = pages_viewed.copy()  # Deep copy
print("\nShallow copy of visitors:\n", visitors_shallow_copy)
print("Deep copy of pages viewed:\n", pages_viewed_deep_copy)
# 3. new concated array!
concatenated_array = np.concatenate((visitors, pages_viewed))
print("\nConcatenated array:\n", concatenated_array)

# 4. Split the concatenated array into two parts based on a threshold (e.g., more than 100 pages viewed).
# Using boolean indexing to separate values above and below the threshold.
threshold = 100
above_threshold = concatenated_array[concatenated_array > threshold]
below_threshold = concatenated_array[concatenated_array <= threshold]

print("\nArray with values above the threshold of 100:\n", above_threshold)
print("Array with values at or below the threshold of 100:\n", below_threshold)

# 5.modifya dn check shallow and deep copy
visitors[0] = 150
pages_viewed[1] = 180

print("\nModified original visitors array:\n", visitors)
print("Modified original pages viewed array:\n", pages_viewed)
print("\nShallow copy of visitors after modification:\n", visitors_shallow_copy)
print("Deep copy of pages viewed after modification:\n", pages_viewed_deep_copy)