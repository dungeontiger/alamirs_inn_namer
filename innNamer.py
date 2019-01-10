import random

patterns1 = [
  'Inn of the {} {}',
  'The {} {} Inn',
  'Tavern of the {} {}',
  'The {} {} Tavern',
  'Sign of the {} {}',
  'The {} {}'
]

patterns2 = [
  "The {}'s {} Inn",
  "The {}'s {} Tavern"
]

with open ('subjects.txt') as f:
  subjects = f.read().split('\n')

with open ('adjectives.txt') as f:
  adjectives = f.read().split('\n')

with open ('objects.txt') as f:
  objects = f.read().split('\n')

i = 10000
while i > 0:
  i = i - 1
  if random.randint(1,2) == 1:
    # Inn of the Red Keep
    # Red Keep Inn
    print (random.choice(patterns1).format(random.choice(adjectives).capitalize(), random.choice(subjects).capitalize()))
  else:
    # The General's Pride Inn
    print (random.choice(patterns2).format(random.choice(subjects).capitalize(), random.choice(objects).capitalize()))