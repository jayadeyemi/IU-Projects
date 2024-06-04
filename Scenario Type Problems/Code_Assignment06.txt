def minWindow(s, t):
    from collections import Counter
    
    if not t or not s:
        return ""
    
    target_char_count = Counter(t)
    required_unique_chars = len(target_char_count)
    
    filtered_s = [(i, char) for i, char in enumerate(s) if char in target_char_count]
    
    left, right = 0, 0
    formed_matches = 0
    window_char_counts = {}
    min_window = float("inf"), None, None
    
    while right < len(filtered_s):
        current_char = filtered_s[right][1]
        window_char_counts[current_char] = window_char_counts.get(current_char, 0) + 1
        
        if window_char_counts[current_char] == target_char_count[current_char]:
            formed_matches += 1
        
        while left <= right and formed_matches == required_unique_chars:
            start_char = filtered_s[left][1]
            
            window_end = filtered_s[right][0]
            window_start = filtered_s[left][0]
            if window_end - window_start + 1 < min_window[0]:
                min_window = (window_end - window_start + 1, window_start, window_end)
            
            window_char_counts[start_char] -= 1
            if window_char_counts[start_char] < target_char_count[start_char]:
                formed_matches -= 1
            
            left += 1    
        
        right += 1    
    
    return "" if min_window[0] == float("inf") else s[min_window[1]:min_window[2] + 1]


def longestConsecutive(nums):
    if not nums:
        return 0

    num_set = set(nums)
    max_sequence_length = 0

    for num in nums:
        if num - 1 not in num_set:
            current_num = num
            current_length = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            max_sequence_length = max(max_sequence_length, current_length)

    return max_sequence_length


class Instagram:
    def __init__(self):
        self.user_photos = {}
        self.user_follows = {}
        self.current_timestamp = 0

    def sharePhoto(self, user_id, photo_id):
        if user_id not in self.user_photos:
            self.user_photos[user_id] = []
        self.user_photos[user_id].append((self.current_timestamp, photo_id))
        self.current_timestamp += 1

    def getFeed(self, user_id):
        feed = []
        if user_id in self.user_photos:
            feed.extend(self.user_photos[user_id][-10:])
        if user_id in self.user_follows:
            for followed_user_id in self.user_follows[user_id]:
                if followed_user_id in self.user_photos:
                    feed.extend(self.user_photos[followed_user_id][-10:])
        feed.sort(reverse=True, key=lambda x: x[0])
        return [photo_id for _, photo_id in feed[:10]]

    def follow(self, follower_id, followee_id):
        if follower_id not in self.user_follows:
            self.user_follows[follower_id] = set()
        self.user_follows[follower_id].add(followee_id)

    def unfollow(self, follower_id, followee_id):
        if follower_id in self.user_follows and followee_id in self.user_follows[follower_id]:
            self.user_follows[follower_id].remove(followee_id)

def uniqSubstr(ipStr: str) -> list:
    # Dictionary to store the last occurrence of each character
    last_occurrence = {}
    for index, char in enumerate(ipStr):
        last_occurrence[char] = index

    # List to store the lengths of the substrings
    lengths = []
    end = 0
    start = 0

    # Iterate through the string to determine the points at which to cut the substrings
    for index, char in enumerate(ipStr):
        # Update the end to the farthest last occurrence of the characters in the current substring
        end = max(end, last_occurrence[char])
        
        # When the current index matches the end, all characters up to this point can be safely cut
        if index == end:
            lengths.append(end - start + 1)
            start = index + 1

    return lengths

def countSum(nums, target_sum):
    cumulative_sum = 0
    sum_frequencies = {0: 1}
    subarray_count = 0

    for num in nums:
        cumulative_sum += num
        if cumulative_sum - target_sum in sum_frequencies:
            subarray_count += sum_frequencies[cumulative_sum - target_sum]
        sum_frequencies[cumulative_sum] = sum_frequencies.get(cumulative_sum, 0) + 1

    return subarray_count




