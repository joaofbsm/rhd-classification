"""
RHD data cleansing and video correlation with diagnostics
"""

import argparse
import os
from shutil import copyfile

import pandas as pd
from tqdm import tqdm


def get_provar_video_data(input_path):
    bh_data_path = input_path + '/provar_data/VSCAN Belo Horizonte/Archive'
    moc_boc_data_paths = [
        input_path + '/provar_data/VSCAN MOC BOC V1',
        input_path + '/provar_data/VSCAN MOC BOC V2',
        input_path + '/provar_data/VSCAN MOC BOC V3'
    ]

    video_data = []

    # BH data
    for exam_folder in tqdm(os.listdir(bh_data_path)):
        if os.path.isdir(f'{bh_data_path}/{exam_folder}'):
            vscan_id = f'BH_{int(exam_folder.split("_")[1])}'

            for file_name in os.listdir(f'{bh_data_path}/{exam_folder}'):
                if file_name.endswith('.MP4') and file_name.split('_')[1] == exam_folder.split('_')[1]:
                    video_id = file_name[:-4]
                    video_path = f'{bh_data_path}/{exam_folder}/{file_name}'
                    
                    video_data.append((vscan_id, video_id, video_path))
    # MOC BOC data
    for moc_boc_folder in moc_boc_data_paths:
        for date_folder in tqdm(os.listdir(moc_boc_folder)):
            if os.path.isdir(f'{moc_boc_folder}/{date_folder}'):
                for exam_folder in os.listdir(f'{moc_boc_folder}/{date_folder}/Archive/'):
                    # Drop videos from 2017 and 2018 to avoid conflicts in VSCAN IDs
                    if os.path.isdir(f'{moc_boc_folder}/{date_folder}/Archive/{exam_folder}') \
                    and not (exam_folder.split('_')[2].startswith('2018') or exam_folder.split('_')[2].startswith('2017')):
                        vscan_id = f'{moc_boc_folder.split()[-1]}_{int(exam_folder.split("_")[1])}'

                        for file_name in os.listdir(f'{moc_boc_folder}/{date_folder}/Archive/{exam_folder}'):
                            if file_name.endswith('.MP4') and file_name.split('_')[1] == exam_folder.split('_')[1]:
                                video_id = file_name[:-4]
                                video_path = f'{moc_boc_folder}/{date_folder}/Archive/{exam_folder}/{file_name}'

                                video_data.append((vscan_id, video_id, video_path))


    video_data = pd.DataFrame(video_data, columns=['VSCAN ID', 'Video ID', 'Video Path'])

    return video_data


def get_bh_ids(input_path):
    bh2015 = pd.read_excel(f'{input_path}/sheets/Belo Horizonte VSCAN Key 2015.xlsx', sheet_name=None)
    bh2016 = pd.read_excel(f'{input_path}/sheets/Belo Horizonte VSCAN - 2016.xlsx', sheet_name=None)

    # Filter columns
    for k in bh2015.keys():
        bh2015[k] = bh2015[k][['Número no VSCAN', 'Número do paciente no PROVAR']]

    for k in bh2016.keys():
        if 'Número no VSCAN' not in bh2016[k].columns:
            bh2016[k] = bh2016[k].rename(index=str, columns={'Número no Vscan': 'Número no VSCAN'})
        
        bh2016[k] = bh2016[k][['Número no VSCAN', 'Número do paciente no PROVAR']] 
        
    bh_ids = pd.concat(list(bh2015.values()) + list(bh2016.values()))

    # Add BH_ prefix to the VSCAN_ID
    bh_ids['Número no VSCAN'] = 'BH_' + bh_ids['Número no VSCAN'].astype(str)
    # Remove BH from the PROVAR ID
    bh_ids['Número do paciente no PROVAR'] = bh_ids['Número do paciente no PROVAR'].str.strip('BH').astype(str)

    bh_ids = bh_ids.rename(index=str, columns={'Número no VSCAN': 'VSCAN ID', 'Número do paciente no PROVAR': 'PROVAR ID'})

    # Remove cases where it is impossible to define which is the correct folder for each of the diagnostics
    # as even cross checking with the diagnostics sheet don't give a hint of where to find the exams.
    bh_ids = bh_ids[~bh_ids['VSCAN ID'].isin(bh_ids[bh_ids.duplicated(subset='VSCAN ID')]['VSCAN ID'])]

    return bh_ids


def get_moc_boc_ids(input_path):
    moc_ids = pd.read_excel(f'{input_path}/sheets/MOC - PLANILHA COMPLETA FINAL  13-12-2016.xlsx', sheet_name='Plan1')
    redcap_ids = pd.read_excel(f'{input_path}/sheets/PLANILHA REDCAP FINAL OK Camila  07-01-2016.xlsx', sheet_name='Plan1')

    column_filter = ['ID', 'ID VSCAN', 'VSCAN']
    moc_boc_ids = pd.concat([moc_ids[column_filter], redcap_ids[column_filter]])

    moc_boc_ids = moc_boc_ids.dropna()
    moc_boc_ids['ID VSCAN'] = moc_boc_ids['ID VSCAN'].astype(str)

    # Some PROVAR IDs have more than one VSCAN ID. Drop all of them to prevent problematic data.
    moc_boc_ids = moc_boc_ids[~((moc_boc_ids['ID VSCAN'].str.contains(' ', regex=False)) | (moc_boc_ids['ID VSCAN'].str.contains('/', regex=False)))]

    moc_boc_ids = pd.DataFrame({'VSCAN ID': moc_boc_ids['VSCAN'] + '_' + moc_boc_ids['ID VSCAN'], 'PROVAR ID': moc_boc_ids['ID'].astype(str)})

    # Remove cases where it is impossible to define which is the correct folder for each of the diagnostics
    moc_boc_ids = moc_boc_ids.drop_duplicates('VSCAN ID', keep=False)

    return moc_boc_ids


def get_diagnoses_by_provar_id(input_path):
    bh_ids = get_bh_ids(input_path)
    moc_boc_ids = get_moc_boc_ids(input_path)
    final_ids = pd.concat([bh_ids, moc_boc_ids])

    diagnoses = pd.read_excel(f'{input_path}/sheets/ProgramaDeRastreamen_DATA_LABELS_2018-03-18_2143.xls')
    diagnoses = diagnoses[['Record ID', 'Diagnosis']]
    diagnoses = diagnoses.dropna()
    diagnoses = diagnoses[diagnoses['Diagnosis'] != 'Other']
    diagnoses = diagnoses.rename(index=str, columns={'Record ID': 'PROVAR ID'})

    # Remove every non-digit character from the PROVAR ID
    diagnoses['PROVAR ID'] = diagnoses['PROVAR ID'].astype(str).str.replace(r'[a-zA-Z]','')

    # Remove the duplicates that are created after we remove the characters, keeping the first occurence
    # The sheet was thoroughly analysed to ensure that no information is lost in this process
    diagnoses = diagnoses.drop_duplicates(subset='PROVAR ID', keep='first')
    
    final_diagnoses = final_ids.merge(diagnoses, how='left', on='PROVAR ID').dropna()

    return final_diagnoses


def copy_video_files(diagnoses_per_video, output_path):
    os.makedirs(output_path, exist_ok=True)      

    for _, row in tqdm(diagnoses_per_video.iterrows(), total=diagnoses_per_video.shape[0]):
        copyfile(row['Video Path'], f'{output_path}/{row["Video ID"]}.MP4')


def cleanse_provar_dataset(input_path, output_path):
    print('Cleansing PROVAR dataset')
    print('========================')

    video_data = get_provar_video_data(input_path)

    final_diagnoses = get_diagnoses_by_provar_id(input_path)

    diagnoses_per_video = video_data.merge(final_diagnoses, how='left', on='VSCAN ID').dropna()
    diagnoses_per_video = diagnoses_per_video.rename(index=str, columns={'VSCAN ID': 'Exam ID'})
    diagnoses_per_video = diagnoses_per_video.sort_values(by='Video Path')
    diagnoses_per_video = diagnoses_per_video.drop_duplicates(subset='Video ID', keep='first')

    copy_video_files(diagnoses_per_video, output_path)

    diagnoses_per_video['Source Dataset'] = 'PROVAR'

    return diagnoses_per_video[['Exam ID', 'Video ID', 'Diagnosis', 'Source Dataset']]


def get_craig_video_data(input_path):
    data_folders = [
        f'{input_path}/craig_data/Group A Full',
        f'{input_path}/craig_data/Group B Full',
        f'{input_path}/craig_data/Group C Full'
    ]

    video_data = []

    for data_folder in tqdm(data_folders):
        for sub_folder in os.listdir(data_folder):
            for exam_folder in os.listdir(f'{data_folder}/{sub_folder}/Archive'):
                for f in os.listdir(f'{data_folder}/{sub_folder}/Archive/{exam_folder}'):
                    if f.endswith('.MP4'):
                        exam_id = f'{exam_folder.split("_")[0][-2:]}_{int(exam_folder.split("_")[1])}'
                        video_id = f[:-4]
                        video_path = f'{data_folder}/{sub_folder}/Archive/{exam_folder}/{f}'
                        
                        video_data.append((exam_id, video_id, video_path))
                        
    video_data = pd.DataFrame(video_data, columns=['Exam ID', 'Video ID', 'Video Path'])

    return video_data


def get_diagnoses_by_craig_exam_id(input_path):
    spiked_uganda = pd.read_excel(f'{input_path}/sheets/LAX VSCAN project summary update 3 2 16 9PM.xlsx', sheet_name='Spiked Uganda Summary', usecols=['Final Diagnosis SP', 'study #'])
    spiked_uganda.columns = ['Diagnosis', 'Study Number']
    spiked_uganda['Machine Name'] = '5S'
    spiked_uganda = spiked_uganda[['Machine Name', 'Study Number', 'Diagnosis']]

    gulu_2013 = pd.read_excel(f'{input_path}/sheets/LAX VSCAN project summary update 3 2 16 9PM.xlsx', sheet_name='GULU 2013 Summary', usecols=['Final Diagnosis SP', 'machine', 'study #'])
    gulu_2013.columns = ['Diagnosis', 'Machine Name', 'Study Number']
    gulu_2013 = gulu_2013[['Machine Name', 'Study Number', 'Diagnosis']]

    mp_nurse = pd.read_excel(f'{input_path}/sheets/LAX VSCAN project summary update 3 2 16 9PM.xlsx', sheet_name='MP NURSE SUMMARY', usecols=['Diagnosis', 'Machine', 'Number'])
    mp_nurse.columns = ['Diagnosis', 'Machine Name', 'Study Number']
    mp_nurse = mp_nurse[['Machine Name', 'Study Number', 'Diagnosis']]
    mp_nurse['Machine Name'] = mp_nurse['Machine Name'].str[-2:] 

    belo_may = pd.read_excel(f'{input_path}/sheets/LAX VSCAN project summary update 3 2 16 9PM.xlsx', sheet_name='Belo May 2015 Summary', usecols=['Diagnosis', 'VSCAN Device S/N', 'VScan number'])
    belo_may.columns = ['Diagnosis', 'Machine Name', 'Study Number']
    belo_may = belo_may[['Machine Name', 'Study Number', 'Diagnosis']]

    final_diagnoses = pd.concat([spiked_uganda, gulu_2013, mp_nurse, belo_may])
    final_diagnoses['Exam ID'] = final_diagnoses['Machine Name'] + '_' + final_diagnoses['Study Number'].astype('str')
    final_diagnoses = final_diagnoses[['Exam ID', 'Diagnosis']]

    # Remove exam with problematic diagnosis
    final_diagnoses = final_diagnoses[final_diagnoses['Exam ID'] != 'SX_326']

    final_diagnoses = final_diagnoses.drop_duplicates(subset='Exam ID')

    final_diagnoses['Diagnosis'] = final_diagnoses['Diagnosis'].replace({'Borderline': 'Borderline RHD', 'Definite': 'Definite RHD'})

    return final_diagnoses


def cleanse_craig_dataset(input_path, output_path):
    print('\nCleansing Craig dataset')
    print('=======================')

    video_data = get_craig_video_data(input_path)

    final_diagnoses = get_diagnoses_by_craig_exam_id(input_path)

    diagnoses_per_video = video_data.merge(final_diagnoses, how='left', on='Exam ID').dropna()
    diagnoses_per_video = diagnoses_per_video.drop_duplicates(subset='Video ID', keep='first')

    copy_video_files(diagnoses_per_video, output_path)

    diagnoses_per_video['Source Dataset'] = 'Craig'

    return diagnoses_per_video[['Exam ID', 'Video ID', 'Diagnosis', 'Source Dataset']]

def main(args):
    diagnoses_per_video = cleanse_provar_dataset(args.input_path, args.output_path)

    if args.craig_present:
        craig_diagnoses_per_video = cleanse_craig_dataset(args.input_path, args.output_path)
        diagnoses_per_video = pd.concat([diagnoses_per_video, craig_diagnoses_per_video])

    diagnoses_per_video.to_csv(f'{args.output_path}/../cleansed_rhd_info.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cleanse datasets of RHD videos.')

    parser.add_argument('-i', '--input-path', type=str, dest='input_path',
                        help='Input path where the PROVAR and Craig data can be found.')
    parser.add_argument('-o', '--output-path', type=str, dest='output_path',
                        help='Output path to which the cleansed data must be saved.')
    parser.add_argument('-c', '--craig-present', action='store_true', default=False, dest='craig_present',
                        help='Flag to indicate if Craig data is present and should be cleansed.')                     

    main(parser.parse_args())
