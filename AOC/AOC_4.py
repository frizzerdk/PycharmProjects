def fully_contains(pair1, pair2):
  # Find the dash in the pair strings
  pair1_dash = pair1.find("-")
  pair2_dash = pair2.find("-")
  print("pair_set: "+pair1+" "+pair2)
  # Use the slice() method to split the strings into their individual numbers
  pair1 = [int(pair1[:pair1_dash]), int(pair1[pair1_dash + 1:])]
  pair2 = [int(pair2[:pair2_dash]), int(pair2[pair2_dash + 1:])]

  # Convert the range objects to set objects
  pair1_set = set(range(pair1[0], pair1[1] +1))
  pair2_set = set(range(pair2[0], pair2[1] +1))

  print(pair1_set , pair2_set)
  # Check if the range of the first pair fully contains the range of the second pair
  check=pair1_set.issuperset(pair2_set) or pair2_set.issuperset(pair1_set)
  if(check):
      print("contained")

  check2 = pair1_set.intersection(pair2_set) != set()
  if (check2):
      print("overlap")

  return check, check2

def count_fully_contained_pairs(pairs_string):
  # Split the input string into individual pairs
  pairs = pairs_string.split("\n")
  print("pairs")
  print(pairs)
  # Start a counter
  counter1 = 0
  counter2 = 0


  # Loop through the pairs
  for i in range(len(pairs)):
      # Split each pair into its individual numbers
      print(i)
      pair1 = pairs[i].split(",")[0]
      pair2 = pairs[i].split(",")[1]


      # Compare the pairs
      check1,check2=fully_contains(pair2, pair1)
      print("check:",check1,check2)
      if check1:
        # Increment the counter if one pair fully contains the other
        counter1 += 1
      if check2:
        # Increment the counter if one pair fully contains the other
        counter2 += 1



  # Return the counter
  return counter1, counter2

# Define the input
with open("AOC_data_4", "r") as f:
  # Read the contents of the file into a string
  text = f.read().strip()

pairs_string = """2-4,6-8
2-3,4-5
5-7,7-9
2-8,3-7
6-6,4-6
2-6,4-8"""
pairs_string=text

# Solve the problem
result1, result2 = count_fully_contained_pairs(pairs_string)

print("result",result1,result2)  # Expected output: 2
