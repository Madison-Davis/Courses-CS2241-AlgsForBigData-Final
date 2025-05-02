# pcrb.py


# ++++++++++++ Imports and Installs ++++++++++++ #
import heapq


# ++++++++++++++ Global Variables ++++++++++++++ #


# ++++++++++++++ Class Definitions ++++++++++++++ #
class PrioritizedCacheReplayBuffer:
    def __init__(self, tier_sizes, quantize_ranges, sampling_weights):
        """
        Initialize class
        """
        assert len(tier_sizes) == 3
        self.tiers = [[] for _ in range(len(tier_sizes))]     # set up tiers as min-heaps using heaq
        self.tier_sizes = tier_sizes                          # how many items can each tier hold (ex: [1, 3, 10])
        self.quantize_ranges = quantize_ranges                # list of (min, max) tuples for obs, nextobs, and actions
        self.sampling_weights = sampling_weights              # when sampling, this is the 'weight' for selecting from each tier

    def insert(self, data):
        """
        Insert a data item (td error, data) into the correct tier and position based on TD error.
        Uses 'trickle-down' logic: if a more surprising point comes in,
        it percolates down less surprising ones. Quantization happens as we go down.
       
        current_data: The current value we're dealing with, either the 'data' arg or data we evicted
        and need to percolate down.  We insert data into heaps, and the td_error serves as the data's key. 
        current_data_tier: The tier number our current data is coming from.  starts at tier 1.
        if, for example, we're dealing with data that comes from tier 2 and we put it in tier 3, needs
        to be quantized 1 level deeper
        """
        td_error = data[-1]
        current_data = (td_error, data)
        current_data_tier = 1               
        
        # For each tier (i = 1,2,3,etc.)
        for i in range(1,len(self.tiers)+1):
            # Get this tier's min and max TD error data values
            tier = self.tiers[i-1]
            min_td = tier[0][0]
            max_td = tier[-1][0]
            # If our current tier is not totally full, just put it in
            if len(tier) < self.tier_capacities[i]:
                # Quantize data based on tier level (1,2,3,etc.) it came from vs being inserted into
                current_data = self.quantize(current_data, current_data_tier, i)
                heapq.heappush(tier, current_data)
                return
            # Otherwise, our current tier is full
            # If our current TD falls in between the current tier's min/max, percolate
            elif min_td <= td_error <= max_td:
                # Quantize data based on tier level (1,2,3,etc.) it came from vs being inserted into
                current_data = self.quantize(current_data, current_data_tier, i)
                heapq.heappush(tier, current_data)
                # If the heap exceeds capacity after insertion, evict the least surprising item
                # and deal with it in the next tier (if there is another tier; otherwise it's discarded)
                if len(tier) > tier.capacity:
                    current_data = heapq.heappop(tier)
                    current_data_tier = i
                return
            # If our current TD is outside the current tier's min/max, try a different tier
            # Otherwise, if we've exhuausted all tiers, we discard the item
            
    def quantize(self, current_data, prev_tier, new_tier):
        """
        Quantize obs, nextobs, and actions in 'current_data'.
        Base the quantization off of if it was already quantized ('prev_tier')
        and what tier we're inserting it into ('curre_tier').
        """
        pass

    def sample(self):
        """
        Sample a single experience;
        Proportionally across tiers, uniformly within each tier.
        """
        pass
