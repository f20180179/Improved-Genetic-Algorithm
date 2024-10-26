import random
import matplotlib.pyplot as plt

def fitness8Queens(entity):
    # as each queen is represented such that it is in separate columns,
    # we need to only check for horizontal and diagonal attacking

    horizontal_attacking = 0
    diagonal_attacking = 0

    count_positions_on_rows = [entity.count(q) for q in entity]
    for count in count_positions_on_rows:
        horizontal_attacking += count - 1
    horizontal_attacking = horizontal_attacking / 2

    n = 8
    negative_diagonal = [0 for _ in range(2 * n)]
    positive_diagonal = [0 for _ in range(2 * n)]

    for i in range(n):
        negative_diagonal[i + entity[i] - 1] += 1
        positive_diagonal[n - i + entity[i] - 2] += 1

    for i in range(2 * n - 1):
        counter = 0
        if negative_diagonal[i] > 1:
            counter += negative_diagonal[i] - 1
        if positive_diagonal[i] > 1:
            counter += positive_diagonal[i] - 1
        diagonal_attacking += counter / (n - abs(i - n + 1))

    total_attacking_pairs = horizontal_attacking + diagonal_attacking
    return int(max_fitness - total_attacking_pairs)


def setProbAccToFitness(entity):
    return fitness8Queens(entity) / max_fitness


def mutate8Queens(child):
    new_child = child
    pos = random.randint(0, 7)
    new_row = random.randint(1, 8)
    new_child[pos] = 0
    while new_row in new_child:
        new_row = random.randint(1, 8)
    new_child[pos] = new_row
    if fitness8Queens(new_child) > fitness8Queens(child):
        return new_child
    return child


def doubleMutate8Queens(child):
    child[random.randint(0, 7)] = random.randint(1, 8)
    child[random.randint(0, 7)] = random.randint(1, 8)
    return child


def reProduce8Queens(x, y):
    best_fitness_value = 0
    best_fit_child = None
    for iter in range(10):
        left_pos = random.randint(0, 7)
        right_pos = 0
        if left_pos == 7:
            right_pos = 7
        else:
            right_pos = random.randint(left_pos + 1, 7)

        z = [0 for _ in range(8)]
        part_from_first = x[left_pos:right_pos + 1]

        iter = 0
        for i in range(8):
            if y[i] not in part_from_first and (iter < left_pos or iter > right_pos):
                z[iter] = y[i]
                iter += 1

        z1 = z
        reversed_part_from_first = part_from_first[::-1]

        iter = 0
        for i in range(left_pos, right_pos + 1):
            z[i] = part_from_first[iter]
            z1[i] = reversed_part_from_first[iter]
            iter += 1

        iter = right_pos + 1
        for i in range(8):
            if y[i] not in z[:right_pos + 1] and iter < 8:
                z[iter] = y[i]
                z1[iter] = y[i]
                iter += 1

        if fitness8Queens(z1) > fitness8Queens(z):
            z = z1

        if best_fitness_value < fitness8Queens(z):
            best_fitness_value = fitness8Queens(z)
            best_fit_child = z

    return best_fit_child

def fitnessTSP(entity):
    total_dist = 0
    for i in range(14-1):
        total_dist += distance_matrix[entity[i]][entity[i+1]]
    return 1000000/total_dist

def reProduceTSP(x, y):
    best_fitness_value = 0
    best_fit_child = None
    for iter in range(10):
        left_pos = random.randint(0,13)
        right_pos = 0
        if left_pos == 13:
            right_pos = 13
        else:
            right_pos = random.randint(left_pos+1, 13)

        z = [0 for _ in range(14)]
        part_from_first = x[left_pos:right_pos+1]

        iter = 0
        for i in range(14):
            if y[i] not in part_from_first and (iter < left_pos or iter > right_pos):
                z[iter] = y[i]
                iter += 1

        z1 = z
        reversed_part_from_first = part_from_first[::-1]

        iter = 0
        for i in range(left_pos, right_pos+1):
            z[i] = part_from_first[iter]
            z1[i] = reversed_part_from_first[iter]
            iter += 1

        iter = right_pos + 1
        for i in range(14):
            if y[i] not in z[:right_pos+1] and iter < 14:
                z[iter] = y[i]
                z1[iter] = y[i]
                iter += 1

        if fitnessTSP(z1) > fitnessTSP(z):
            z = z1

        if best_fitness_value < fitnessTSP(z):
            best_fitness_value = fitnessTSP(z)
            best_fit_child = z

    return best_fit_child


def mutateTSP(child):
    pos1 = random.randint(0, 13)
    pos2 = random.randint(0, 13)
    new_child = child
    new_child[pos1], new_child[pos2] = new_child[pos2], new_child[pos1]
    if fitnessTSP(new_child) > fitnessTSP(child):
        return new_child
    return child

def doubleMutateTSP(child):
    new_child = child
    pos1 = random.randint(0, 13)
    pos2 = random.randint(0, 13)
    new_child[pos1], new_child[pos2] = new_child[pos2], new_child[pos1]
    pos1 = random.randint(0, 13)
    pos2 = random.randint(0, 13)
    new_child[pos1], new_child[pos2] = new_child[pos2], new_child[pos1]
    if fitnessTSP(new_child) > fitnessTSP(child):
        return new_child
    return child

def findFitnessForEachEntityTSP(population):
    fitness_values = []
    for entity in population:
        fitness_values.append(fitnessTSP(entity))
    return fitness_values

def findFittestEntityTSP(fitness_for_each_entity):
    max_fitness = 0
    best_fit_entity = 0
    for k,v in enumerate(fitness_for_each_entity):
        if max_fitness < v:
            best_fit_entity = k
            max_fitness = v
    return best_fit_entity

def fitnessStoppedChangingTSP(past_10_generations):
    min_fitness = 10000
    max_fitness = 0
    for i in range(10):
        min_fitness = min(min_fitness, past_10_generations[i])
        max_fitness = max(max_fitness, past_10_generations[i])
    return max_fitness-min_fitness

if __name__ == "__main__":
    print("Press 1 to run 8 queens problem.")
    print("Press 2 to run Travelling Salesman problem")
    choice = int(input())
    if choice == 1:
        n_runs = 0
        number_of_generations_taken = []
        for n_runs in range(1):
            max_fitness = 1 + 28
            initial_board = [1 for i in range(8)]
            population_size = 20
            population = [initial_board for _ in range(population_size)]
            n_iter = 1
            max_fitness_so_far = 0
            max_fitness_per_generations = []

            # Till some fit individual is not found
            while max_fitness_so_far < max_fitness:  # (not max_fitness in [fitness(entities) for entities in population]):
                new_population = []
                max_fitness_in_this_gen = 0
                prob_acc_to_fitness = [setProbAccToFitness(entity) for entity in population]

                for i in range(len(population)):
                    x = random.choices(population, weights=prob_acc_to_fitness, k=1)
                    y = random.choices(population, weights=prob_acc_to_fitness, k=1)
                    child = reProduce8Queens(x[0], y[0])
                    # if small prob, mutate the child
                    prob_of_mutation = random.random()
                    if (prob_of_mutation < 0.02):
                        child = doubleMutate8Queens(child)
                    if (prob_of_mutation < 0.03):
                        child = mutate8Queens(child)

                    max_fitness_in_this_gen = max(max_fitness_in_this_gen, fitness8Queens(child))
                    max_fitness_so_far = max(max_fitness_so_far, max_fitness_in_this_gen)

                    new_population.append(child)

                max_fitness_per_generations.append(max_fitness_in_this_gen)
                population = new_population
                print("Best Fitness value in {}th generation : ".format(n_iter), max_fitness_in_this_gen)
                n_iter += 1

            number_of_generations_taken.append(n_iter)

            print("Found best fitness in {} iterations: ".format(n_iter-1))
            for entity in population:
                if fitness8Queens(entity) == max_fitness:
                    print(entity)
                    break

            x = [i for i in range(n_iter - 1)]
            y = max_fitness_per_generations

            plt.plot(x, y)
            plt.xlabel('Number of Genarations')
            plt.ylabel('Fitness value')
            plt.show()

        # print("On average for 100 runs, took {} generations.".format(sum(number_of_generations_taken)/100))
        # print("On minimum for 100 runs, took {} generations.".format(min(number_of_generations_taken)))
        # print("On maximum for 100 runs, took {} generations.".format(max(number_of_generations_taken)))
    else:
        number_of_generations_taken = []
        for n_runs in range(1):
            population_size = 20
            initial_path = [i for i in range(14)]
            population = [initial_path for _ in range(population_size)]

            distance_matrix = [[10000 for _ in range(14)] for __ in range(14)]
            for i in range(14):
                for j in range(14):
                    if i == j:
                        distance_matrix[i][j] = 0

            distance_matrix[0][6] = 150
            distance_matrix[0][9] = 200
            distance_matrix[0][11] = 120
            distance_matrix[1][7] = 190
            distance_matrix[1][8] = 400
            distance_matrix[1][13] = 130
            distance_matrix[2][3] = 600
            distance_matrix[2][4] = 220
            distance_matrix[2][5] = 400
            distance_matrix[2][8] = 200
            distance_matrix[3][2] = 600
            distance_matrix[3][5] = 210
            distance_matrix[3][10] = 300
            distance_matrix[4][2] = 220
            distance_matrix[4][8] = 180
            distance_matrix[5][2] = 400
            distance_matrix[5][3] = 210
            distance_matrix[5][10] = 370
            distance_matrix[5][11] = 600
            distance_matrix[5][12] = 260
            distance_matrix[5][13] = 900
            distance_matrix[6][0] = 150
            distance_matrix[6][10] = 550
            distance_matrix[6][11] = 180
            distance_matrix[7][1] = 190
            distance_matrix[7][9] = 560
            distance_matrix[7][13] = 170
            distance_matrix[8][1] = 400
            distance_matrix[8][2] = 200
            distance_matrix[8][4] = 180
            distance_matrix[8][13] = 600
            distance_matrix[9][0] = 200
            distance_matrix[9][7] = 560
            distance_matrix[9][11] = 160
            distance_matrix[9][13] = 500
            distance_matrix[10][3] = 300
            distance_matrix[10][5] = 370
            distance_matrix[10][6] = 550
            distance_matrix[10][12] = 240
            distance_matrix[11][0] = 120
            distance_matrix[11][5] = 600
            distance_matrix[11][6] = 180
            distance_matrix[11][9] = 160
            distance_matrix[11][12] = 400
            distance_matrix[12][5] = 260
            distance_matrix[12][10] = 240
            distance_matrix[12][11] = 400
            distance_matrix[13][1] = 130
            distance_matrix[13][5] = 900
            distance_matrix[13][7] = 170
            distance_matrix[13][8] = 600
            distance_matrix[13][9] = 500

            max_fitness = 0
            n_generations = 0
            max_fitness_per_generation = []
            fitness_for_each_entity = findFitnessForEachEntityTSP(population)
            max_fitness_value = max(fitness_for_each_entity)
            best_entity = findFittestEntityTSP(fitness_for_each_entity)
            distance_covered = 1000000 * (1 / fitness_for_each_entity[best_entity])
            max_fitness_per_generation = []
            past_10_generations = []

            while 1:  # distance_covered >= 3175:
                if n_generations > 10:
                    if fitnessStoppedChangingTSP(past_10_generations) <= 0.00000001:
                        break
                new_population = []
                total_fitness = sum(fitness_for_each_entity)
                set_prob_acc_to_fitness = [fitness_for_each_entity[i] / total_fitness for i in range(len(population))]
                for i in range(len(population)):
                    x = random.choices(population, weights=set_prob_acc_to_fitness, k=1)
                    y = random.choices(population, weights=set_prob_acc_to_fitness, k=1)
                    child = reProduceTSP(x[0], y[0])
                    if random.random() < 0.02:
                        child = doubleMutateTSP(child)
                    if random.random() < 0.03:
                        child = mutateTSP(child)

                    new_population.append(child)

                n_generations += 1
                population = new_population
                fitness_for_each_entity = findFitnessForEachEntityTSP(population)
                max_fitness_value = max(fitness_for_each_entity)
                max_fitness_per_generation.append(max_fitness_value)
                best_entity = findFittestEntityTSP(fitness_for_each_entity)
                distance_covered = 1000000 * (1 / fitness_for_each_entity[best_entity])
                if n_generations > 10:
                    past_10_generations.pop(0)
                past_10_generations.append(max_fitness_value)
                print("Best fitness in {}th generation : ".format(n_generations), max_fitness_value)

            number_of_generations_taken.append(n_generations)

            best_entity = findFittestEntityTSP(fitness_for_each_entity)
            print("Fitness started to become constant after {} generations ".format(n_generations))
            print("Best fitness value: ", fitness_for_each_entity[best_entity])
            print("Distance covered: ", 1000000 * (1 / fitness_for_each_entity[best_entity]))
            print("Best entity: ", population[best_entity])

            x = [i for i in range(n_generations)]
            y = max_fitness_per_generation

            plt.plot(x, y)
            plt.xlabel('Number of Generations')
            plt.ylabel('Fitness value')
            plt.show()

        # print("On average for 100 runs, took {} generations.".format(sum(number_of_generations_taken)/100))
        # print("On minimum for 100 runs, took {} generations.".format(min(number_of_generations_taken)))
        # print("On maximum for 100 runs, took {} generations.".format(max(number_of_generations_taken)))
    exit(0)
