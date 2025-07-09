# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src import optimizer

def load_data_from_uploads(uploaded_appointments, uploaded_patients, uploaded_slots):
    """Load data from the user's file uploads. Returns dataframes if all are provided."""
    if uploaded_appointments and uploaded_patients and uploaded_slots:
        appointments = pd.read_csv(uploaded_appointments)
        patients = pd.read_csv(uploaded_patients)
        slots = pd.read_csv(uploaded_slots)
        return appointments, patients, slots
    return None, None, None

@st.cache(allow_output_mutation=True)
def run_optimization(appointments_file, patients_file, slots_file, doctors_input):
    # If user has uploaded new files, use them. Otherwise, use the defaults from /data/.
    if appointments_file and patients_file and slots_file:
        # Save temporary uploaded files and process them directly
        appointments_file.seek(0)
        patients_file.seek(0)
        slots_file.seek(0)
        appointments = pd.read_csv(appointments_file)
        patients = pd.read_csv(patients_file)
        slots = pd.read_csv(slots_file)
        # Save them temporarily in memory to a dataframe
        # Standardize columns and preprocessing steps:
        for df in [appointments, patients, slots]:
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            df.drop_duplicates(inplace=True)
        # Merge the dataframes the same way as in our static pipeline:
        if ('scheduled_day' in appointments.columns and 'appointment_day' in appointments.columns):
            appointments['scheduled_day'] = pd.to_datetime(appointments['scheduled_day'])
            appointments['appointment_day'] = pd.to_datetime(appointments['appointment_day'])
            appointments['waiting_days'] = (appointments['appointment_day'] - appointments['scheduled_day']).dt.days
        df_merged = pd.merge(appointments, patients, on='patient_id', how='left')
        df_merged = pd.merge(df_merged, slots, on='slot_id', how='left')
    else:
        df_merged = optimizer.load_and_preprocess_data()  # default static files

    # Setup scheduling parameters with dynamic input doctor list
    if doctors_input:
        doctors = [doc.strip() for doc in doctors_input.split(',') if doc.strip()]
    else:
        doctors = None  # use default if none entered

    scheduling_params = optimizer.setup_scheduling_problem(df_merged, doctors_input=doctors)
    best_solution, best_fitness = optimizer.koa_optimization(
        avg_workload=scheduling_params['avg_workload'],
        num_doctors=scheduling_params['num_doctors'],
        num_slots=scheduling_params['num_slots'],
        max_slots=scheduling_params['max_slots']
    )
    
    # Optionally simulate emergency re-optimization if the user desires
    reoptimized_solution = optimizer.emergency_reoptimization(
        best_solution,
        avg_workload=scheduling_params['avg_workload'],
        num_doctors=scheduling_params['num_doctors'],
        max_slots=scheduling_params['max_slots'],
        unavailable_doctor=1  # You can also provide an input for which doctor to remove
    )
    
    # Create a final schedule dataframe mapping slot_id to doctor name
    final_schedule = pd.DataFrame({
        'slot_id': scheduling_params['slots_list'],
        'assigned_doctor': [scheduling_params['doctors'][doc_idx] for doc_idx in reoptimized_solution]
    })
    return final_schedule, scheduling_params

def plot_workload(schedule, num_doctors):
    # Count assigned slots per doctor. Assumes doctor names are unique.
    workloads = schedule['assigned_doctor'].value_counts().reindex(
        [f"Dr_{chr(65+i)}" for i in range(num_doctors)], fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(workloads.index, workloads.values, color='skyblue')
    ax.set_xlabel("Doctor")
    ax.set_ylabel("Number of Slots")
    ax.set_title("Workload Distribution")
    return fig

def main():
    st.title("AI-Optimized Smart Scheduling Dashboard")
    
    st.sidebar.header("Data & Input Settings")
    st.sidebar.markdown("Upload new CSV files to override the default data or leave blank to use existing files.")
    uploaded_appointments = st.sidebar.file_uploader("Upload appointments.csv", type="csv")
    uploaded_patients = st.sidebar.file_uploader("Upload patients.csv", type="csv")
    uploaded_slots = st.sidebar.file_uploader("Upload slots.csv", type="csv")
    
    doctors_input = st.sidebar.text_input("Enter Doctor List (comma-separated)", value="Dr_A, Dr_B, Dr_C, Dr_D")
    
    if st.sidebar.button("Run Optimization"):
        st.write("Running optimization, please wait...")
        final_schedule, scheduling_params = run_optimization(uploaded_appointments, uploaded_patients, uploaded_slots, doctors_input)
        st.success("Optimization completed!")
        
        st.write("### Final Optimized Schedule")
        st.dataframe(final_schedule)
        
        st.write("### Workload Distribution")
        fig = plot_workload(final_schedule, scheduling_params['num_doctors'])
        st.pyplot(fig)
        
        # Option to download the resulting schedule
        csv = final_schedule.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Final Schedule as CSV",
            data=csv,
            file_name='final_optimized_schedule.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()
