import numpy as np

T = 10

vol_0 = np.array([70, 100, 175, 190])
vol_growth = np.array([0.01, 0.11, 0.18, 0.02])
cost_per_vol = np.array([100, 150, 320, 210])
decisions = np.zeros(T)

money_0 = 95000
interest = 0.075

print(95000 - (175 * 320 + 100 * 150))
print(int(24000 * (1 + interest) ** T))
print(70 * 1.01 ** T * 100)
print(190 * 1.02 ** T * 210)

def solve(
    volumes : np.ndarray,
    money : float,
    choices : np.ndarray,
    time : int
): 

    if time == T:
        # if a volume is still above 0, we get zero money

        if np.sum(choices) != 10:
            return -np.inf, choices, volumes
        else:
            return money, choices, volumes
    
    max_money = -np.inf
    max_choices = np.zeros(T)
    max_volumes = np.zeros(4)

    for choice_index in range(5):

        pit_index = choice_index - 1

        # if pit is empty, we can't empty it again
        if volumes[pit_index] <= 0:
            continue

        current_money = money
        current_volumes = np.copy(volumes)
        current_choices = np.copy(choices)

        # adjust volumes and money for emptying pit
        if choice_index > 0:
            current_choices[time] = choice_index
            current_money = current_money - cost_per_vol[pit_index] * current_volumes[pit_index]
            current_volumes[pit_index] = 0

        current_money = (1 + interest) * current_money
        current_volumes = current_volumes * (1 + vol_growth)
        
        current_money, current_choices, max_volumes = solve(
            current_volumes,
            current_money,
            current_choices,
            time + 1)
        
        if current_money > max_money:
            max_money = current_money
            max_choices = current_choices
            max_volumes = current_volumes
        
    return max_money, max_choices, max_volumes

print(solve(vol_0, money_0, decisions, 0))






    