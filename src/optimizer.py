# src/optimizer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def load_and_preprocess_data(appointments_path='./data/appointments.csv',
                             patients_path='./data/patients.csv',
                             slots_path='./data/slots.csv'):
    """
    Load CSV files from given paths and perform cleaning, merging, and feature engineering.
    """
    appointments = pd.read_csv(appointments_path)
    patients = pd.read_csv(patients_path)
    slots = pd.read_csv(slots_path)

    # Standardize column names for each dataframe
    for df in [appointments, patients, slots]:
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df.drop_duplicates(inplace=True)
    
    # Feature engineering: waiting time conversion, if the columns exist.
    if 'scheduled_day' in appointments.columns and 'appointment_day' in appointments.columns:
        appointments['scheduled_day'] = pd.to_datetime(appointments['scheduled_day'])
        appointments['appointment_day'] = pd.to_datetime(appointments['appointment_day'])
        appointments['waiting_days'] = (appointments['appointment_day'] - appointments['scheduled_day']).dt.days

    # Data merging (left joins)
    df_merged = pd.merge(appointments, patients, on='patient_id', how='left')
    df_merged = pd.merge(df_merged, slots, on='slot_id', how='left')

    # Encode categorical columns – here we encode appointment_status
    if 'appointment_status' not in df_merged.columns:
        df_merged['appointment_status'] = np.nan  # default if missing
    label_encoder = LabelEncoder()
    df_merged['appointment_status_encoded'] = label_encoder.fit_transform(df_merged['appointment_status'].astype(str))

    # Normalize numeric features (if present)
    scaler = MinMaxScaler()
    if 'waiting_days' in df_merged.columns:
        df_merged['waiting_days_norm'] = scaler.fit_transform(df_merged[['waiting_days']])
    
    return df_merged

def setup_scheduling_problem(df_merged, doctors_input=None):
    """
    Prepare optimization problem parameters.
    Users can supply a custom list of doctors.
    """
    # Allow custom doctor list via user input; otherwise, use a default list.
    if doctors_input is None or len(doctors_input) == 0:
        doctors = ['Dr_A', 'Dr_B', 'Dr_C', 'Dr_D']
    else:
        doctors = doctors_input
    
    # Get the list of unique slots from the merged data.
    slots_list = df_merged['slot_id'].unique().tolist()
    num_doctors = len(doctors)
    num_slots = len(slots_list)
    avg_workload = num_slots / num_doctors
    max_slots_per_doctor = int(np.ceil(avg_workload * 1.2))
    
    return {
        'doctors': doctors,
        'slots_list': slots_list,
        'num_doctors': num_doctors,
        'num_slots': num_slots,
        'avg_workload': avg_workload,
        'max_slots': max_slots_per_doctor
    }

def compute_fitness(schedule, avg_workload, num_doctors, max_slots):
    """
    Evaluate the fitness of a candidate schedule. Lower scores are better.
    """
    workloads = np.bincount(schedule, minlength=num_doctors)
    balance_penalty = np.sum(np.abs(workloads - avg_workload))
    constraint_penalty = sum(1000 * (w - max_slots) for w in workloads if w > max_slots)
    return balance_penalty + constraint_penalty

def initialize_population(pop_size, num_slots, num_doctors):
    population = []
    for _ in range(pop_size):
        candidate = np.random.randint(0, num_doctors, size=num_slots)
        population.append(candidate)
    return population

def variation(candidate, variation_rate=0.05):
    new_candidate = candidate.copy()
    num_swaps = int(variation_rate * len(candidate))
    for _ in range(num_swaps):
        idx1, idx2 = np.random.randint(0, len(candidate), 2)
        new_candidate[idx1], new_candidate[idx2] = new_candidate[idx2], new_candidate[idx1]
    return new_candidate

def local_search(candidate, avg_workload, num_doctors, max_slots):
    best_candidate = candidate.copy()
    best_fitness = compute_fitness(best_candidate, avg_workload, num_doctors, max_slots)
    for _ in range(10):
        temp = candidate.copy()
        idx = np.random.randint(0, len(candidate))
        temp[idx] = np.random.randint(0, num_doctors)
        temp_fitness = compute_fitness(temp, avg_workload, num_doctors, max_slots)
        if temp_fitness < best_fitness:
            best_candidate = temp.copy()
            best_fitness = temp_fitness
    return best_candidate

def koa_optimization(avg_workload, num_doctors, num_slots, max_slots, pop_size=50, num_generations=100):
    population = initialize_population(pop_size, num_slots, num_doctors)
    best_solution = None
    best_fitness_overall = np.inf
    
    for generation in range(num_generations):
        new_population = []
        population_fitness = [compute_fitness(candidate, avg_workload, num_doctors, max_slots) for candidate in population]
        idx_best = np.argmin(population_fitness)
        current_best = population[idx_best]
        if population_fitness[idx_best] < best_fitness_overall:
            best_fitness_overall = population_fitness[idx_best]
            best_solution = current_best.copy()
        
        # Exploration: generating candidate variations
        for candidate in population:
            new_population.append(variation(candidate))
        # Exploitation: refine best candidate
        refined = local_search(current_best, avg_workload, num_doctors, max_slots)
        new_population.append(refined)
        # Communication: share best candidate information
        for i in range(len(new_population) // 5):
            idx = np.random.randint(0, len(new_population))
            new_population[idx][:len(best_solution)//2] = best_solution[:len(best_solution)//2]
        population = new_population.copy()
        
        if generation % 10 == 0:
            print(f"Generation {generation}, Best Fitness: {best_fitness_overall}")
    
    print("KOA optimization completed.")
    print("Best overall fitness:", best_fitness_overall)
    return best_solution, best_fitness_overall

def emergency_reoptimization(best_solution, avg_workload, num_doctors, max_slots, unavailable_doctor=1, iterations=50):
    """
    Reassign slots when a specific doctor becomes unavailable.
    """
    affected_slots = np.where(best_solution == unavailable_doctor)[0]
    print("Affected slots count:", len(affected_slots))
    
    new_solution = best_solution.copy()
    for s in affected_slots:
        new_solution[s] = np.random.choice([d for d in range(num_doctors) if d != unavailable_doctor])
    
    for _ in range(iterations):
        idx = np.random.choice(affected_slots)
        temp_solution = new_solution.copy()
        temp_solution[idx] = np.random.randint(0, num_doctors)
        if compute_fitness(temp_solution, avg_workload, num_doctors, max_slots) < compute_fitness(new_solution, avg_workload, num_doctors, max_slots):
            new_solution = temp_solution.copy()
    
    print("Fitness after emergency re-optimization:", compute_fitness(new_solution, avg_workload, num_doctors, max_slots))
    return new_solution
