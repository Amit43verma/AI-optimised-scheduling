# src/main.py
import argparse
import pandas as pd
from src import optimizer

def main(args):
    df = optimizer.load_and_preprocess_data(
        appointments_path=args.appointments,
        patients_path=args.patients,
        slots_path=args.slots
    )
    
    if args.doctors:
        doctors = [d.strip() for d in args.doctors.split(',')]
    else:
        doctors = None
    
    scheduling_params = optimizer.setup_scheduling_problem(df, doctors_input=doctors)
    best_solution, best_fitness = optimizer.koa_optimization(
        avg_workload=scheduling_params['avg_workload'],
        num_doctors=scheduling_params['num_doctors'],
        num_slots=scheduling_params['num_slots'],
        max_slots=scheduling_params['max_slots']
    )
    
    new_solution = optimizer.emergency_reoptimization(
        best_solution,
        avg_workload=scheduling_params['avg_workload'],
        num_doctors=scheduling_params['num_doctors'],
        max_slots=scheduling_params['max_slots'],
        unavailable_doctor=args.unavailable_doctor
    )
    
    final_schedule = pd.DataFrame({
        'slot_id': scheduling_params['slots_list'],
        'assigned_doctor': [scheduling_params['doctors'][doc_idx] for doc_idx in new_solution]
    })
    final_schedule.to_csv(args.output, index=False)
    print(f"Final schedule saved to {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Scheduling Optimization with dynamic input")
    parser.add_argument("--appointments", type=str, default="./data/appointments.csv", help="Path to appointments CSV")
    parser.add_argument("--patients", type=str, default="./data/patients.csv", help="Path to patients CSV")
    parser.add_argument("--slots", type=str, default="./data/slots.csv", help="Path to slots CSV")
    parser.add_argument("--doctors", type=str, default="Dr_A,Dr_B,Dr_C,Dr_D", help="Comma-separated list of doctors")
    parser.add_argument("--unavailable_doctor", type=int, default=1, help="Index of doctor to simulate unavailability")
    parser.add_argument("--output", type=str, default="final_optimized_schedule.csv", help="Output CSV filename")
    args = parser.parse_args()
    main(args)
