import fnmatch
import pandas


def load_config_yml(config_file, individual=False):

    # loads a configuration YAML file
    #
    # input
    #   config_file: full filepath to YAML (.yml) file
    #
    # output
    #   config: Configuration object

    import os
    import yaml
    from CPAC.utils import Configuration

    try:
        config_path = os.path.realpath(config_file)

        with open(config_path,"r") as f:
            config_dict = yaml.load(f)

        config = Configuration(config_dict)

    except Exception as e:
        err = "\n\n[!] CPAC says: Could not load or read the configuration " \
        	  "YAML file:\n%s\nDetails: %s\n\n" % (config_file, e)
        raise Exception(err)

    if individual:
        config.logDirectory = os.path.abspath(config.logDirectory)
        config.workingDirectory = os.path.abspath(config.workingDirectory)
        config.outputDirectory = os.path.abspath(config.outputDirectory)
        config.crashLogDirectory = os.path.abspath(config.crashLogDirectory)

    return config


def load_text_file(filepath, label="file"):

    # loads a text file and returns the lines in a list
    #
    # input
    #   filepath: full filepath to the text file
    #
    # output
    #   lines_list: list of lines from text file

    if not filepath.endswith(".txt"):
        err = "\n\n[!] CPAC says: The %s should be a text file (.txt).\n" \
              "Path provided: %s\n\n" % (label, filepath)
        raise Exception(err)

    try:
        with open(filepath,"r") as f:
            lines_list = f.readlines()
    except Exception as e:
        err = "\n\n[!] CPAC says: Could not load or read the %s:\n%s\n" \
              "Details: %s\n\n" % (label, filepath, e)
        raise Exception(err)

    # get rid of those \n's that love to show up everywhere
    lines_list = [i.rstrip("\n") for i in lines_list]

    return lines_list


def gather_nifti_globs(pipeline_output_folder,resource_list,derivatives=None):
    import os
    import glob
    import pandas as pd
    import pkg_resources as p

    if len(resource_list) == 0:
        err = "\n\n[!] please choose atleast one nusiance stratergy!\n\n"
        raise Exception(err)

    if derivatives is None:

        keys_csv = p.resource_filename('CPAC','resources/cpac_outputs.csv')
        try:
            keys=pd.read_csv(keys_csv)
        except Exception as e:
            err = "\n[!] Could not access or read the cpac_outputs.csv " \
                  "resource file:\n{0}\n\nError details {1}\n".format(keys_csv, e)
            raise Exception(err)
        derivatives = list(
            keys[keys['Derivative'] == 'yes'][keys['Space'] == 'template'][
                keys['Values'] == 'z-score']['Resource'])
        derivatives = derivatives + list(
            keys[keys['Derivative'] == 'yes'][keys['Space'] == 'template'][
                keys['Values'] == 'z-stat']['Resource'])

    pipeline_output_folder = pipeline_output_folder.rstrip("/")
    print "\n\nGathering the output file paths from %s..." \
    % pipeline_output_folder

    search_dir = []
    for derivative_name in derivatives:
        for resource_name in resource_list:
            for resource_name in derivative_name:
                search_dir.append(derivative_name)
    nifti_globs=[]

    for resource_name in search_dir:

        glob_pieces = [pipeline_output_folder, "*", resource_name, "*"]

        glob_query = os.path.join(*glob_pieces)

        found_files = glob.iglob(glob_query)

        exts=['nii','nii.gz']

        still_looking = True
        while still_looking:
            still_looking = False
            for found_file in found_files:

                still_looking = True

                if os.path.isfile(found_file) and any(found_file.endswith('.' + ext) for ext in exts):

                    nifti_globs.append(glob_query)

                    break
            if still_looking:
                glob_query = os.path.join(glob_query, "*")
                found_files = glob.iglob(glob_query)

    if len(nifti_globs) == 0:
        err = "\n\n[!] No output filepaths found in the pipeline output " \
             "directory provided for the derivatives selected!\n\nPipeline " \
             "output directory provided: %s\nDerivatives selected:%s\n\n" \
              % (pipeline_output_folder, resource_list)
        raise Exception(err)

    return nifti_globs,search_dir

def add(x,y):
    a = x+y
    return a

def create_output_dict_list(nifti_globs,pipeline_folder,resource_list,search_dir,derivatives=None):
    import os
    import glob
    import itertools
    import pandas as pd
    import pkg_resources as p

    if len(resource_list) == 0:
        err= "\n\n[!] No derivatives selected!\n\n"
        raise Exception(err)
    if derivatives is None:
        keys_csv = p.resource_filename('CPAC', 'resources/cpac_outputs.csv')
        try:
            keys=pd.read_csv(keys_csv)
        except Exception as e:
            err= "\n[!] Could not access or read the cpac_outputs.csv " \
                "resource file:\n{0}\n\nError details {1}\n".format(keys_csv,e)
            raise Exception(err)
    exts=['nii','nii.gz']
    exts = ['.'+ ext.lstrip('.')for ext in exts]
    output_dict_list={}
    for root,_,files in os.walk(pipeline_folder):
        for filename in files:
            filepath=os.path.join(root,filename)

            if not any(fnmatch.fnmatch(filepath,pattern)for pattern in nifti_globs):
                continue
            if not any(filepath.endswith(ext)for ext in exts):
                continue
            relative_filepath=filepath.split(pipeline_folder)[1]
            filepath_pieces=filter(None, relative_filepath.split("/"))
            resource_id = filepath_pieces[1]

            if resource_id not in search_dir:
                continue

            series_id_string = filepath_pieces[2]
            strat_info = "_".join(filepath_pieces[3:])[:-len(ext)]
            unique_resource_id=(resource_id,strat_info)
            if unique_resource_id not in output_dict_list.keys():
                output_dict_list[unique_resource_id] = []

            unique_id = filepath_pieces[0]

            series_id = series_id_string.replace("_scan_", "")
            series_id = series_id.replace("_rest", "")

            new_row_dict = {}
            new_row_dict["participant_session_id"] = unique_id
            new_row_dict["participant_id"], new_row_dict["Sessions"] = \
                unique_id.split('_')

            new_row_dict["Series"] = series_id
            new_row_dict["Filepath"] = filepath

            print('{0} - {1} - {2}'.format(unique_id, series_id,
                                           resource_id))
            output_dict_list[unique_resource_id].append(new_row_dict)

        #analysis, grouped either by sessions or scans.
    return output_dict_list

def create_output_df_dict(output_dict_list,inclusion_list):

    import pandas as pd

    output_df_dict={}


    for unique_resource_id in output_dict_list.keys():
        ##This dataframe will give you what is in the C-PAC output directory for individual level analysis outputs##
        new_df = pd.DataFrame(output_dict_list[unique_resource_id])
        print(new_df)
        if inclusion_list:
            #this is for participants only, not scans/sessions/etc
            new_df=new_df[new_df.participant_id.isin(inclusion_list)]

        if new_df.empty:
                print("No outputs found for {0} the participants "\
                      "listed in the group manalysis participant list you "\
                      "used. Skipping generating model for this "\
                      "output.".format(unique_resource_id))
                continue
        if unique_resource_id not in output_df_dict.keys():
                output_df_dict[unique_resource_id] = new_df

    return output_df_dict



def gather_outputs(pipeline_folder,resource_list,inclusion_list):



    nifti_globs,search_dir = gather_nifti_globs(pipeline_folder,resource_list,derivatives=None)

    output_dict_list = create_output_dict_list(nifti_globs,pipeline_folder,resource_list,search_dir,derivatives=None)
    # now we have a good dictionary which contains all the filepaths of the files we need to merge later on.
    # Steps after this: 1. This is only a dictionary so let's convert it to a data frame.
    # 2. In the data frame, we're going to only include whatever is in the participant list
    # 3. From the list of included participants, we're going to further prune the output dataframe to only contain the
    # scans included, and/or the sessions included
    # 4. Our final output df will contain, file paths for the .nii files of all the participants that are included in the

    output_df_dict=create_output_df_dict(output_dict_list,inclusion_list)

    return output_df_dict

def prep_inputs(group_config_file):

    import os
    import pandas as pd
    import pkg_resources as p

    keys_csv = p.resource_filename('CPAC','resources/cpac_outputs.csv')
    try:
        keys = pd.read_csv(keys_csv)
    except Exception as e:
        err = "\n[!] Could not access or read the cpac_outputs.csv " \
              "resource file:\n{0}\n\nError details {1}\n".format(keys_csv,e)
        raise Exception(err)

    group_model=load_config_yml(group_config_file)
    pipeline_folder = group_model.pipeline_dir
    #inclusion list function
    if not group_model.participant_list:
        inclusion_list = grab_pipeline_dir_subs(pipeline_folder)
    elif '.' in group_model.participant_list:

        if not os.path.isfile(group_model.participant_list):

            raise Exception('\n[!] C-PAC says: Your participant '
                            'inclusion list is not a valid file!\n\n'
                            'File path: {0}'
                            '\n'.format(group_model.participant_list))
        else:
            inclusion_list = load_text_file(group_model.participant_list,"group-level analysis participant list")

    else:
        inclusion_list = grab_pipeline_dir_subs(pipeline_dir)
    resource_list = ['alff']
    output_df_dict=gather_outputs(pipeline_folder,resource_list,inclusion_list)
    analysis_dict = {}
    for unique_resource in output_df_dict.keys():
        resource_id = unique_resource[0]
    strat_info=unique_resource[1]
    output_df=output_df_dict[unique_resource]
    #We're going to reduce the size of the output df based on nuisance strat and the
    #participant list that actually is included.
    if not group_model.participant_list:
        inclusion_list = grab_pipeline_dir_subs(pipeline_dir)
        output_df = output_df[output_df["participant_id"].isin(inclusion_list)]
    elif os.path.isfile(group_model.participant_list):
        inclusion_list = load_text_file(group_model.participant_list,
                                        "group-level analysis "
                                        "participant list")
        output_df = output_df[output_df["participant_id"].isin(inclusion_list)]
    else:
        raise Exception('\nCannot read group-level analysis participant ' \
                        'list.\n')
    # We're then going to reduce the Output directory to contain only those scans and or the sessions which are expressed by the user.
    # If the user answers all to the option, then we're obviously not going to do any repeated measures.

    repeated_sessions = False
    repeated_scan = False
    repeated_measures=False
    
    if len(group_config_file.qpp_sess_list) > 0:
        repeated_sessions = True
    if len(group_config_file.qpp_scan_list) > 0:
        repeated_scans = True
    if repeated_sessions or repeated_series:
        repeated_measures=True

    if repeated_scans:
        #In this case, you're basically going to curse at yourself for getting it wrong
        #But if a user has multiple scans and want to include all scans in one model, you have
        #to group by sessions
        series = "repeated_measures_multiple_series"
        if 'session' in output_df



def balance_df(output_df,qpp_sess_list):
    import pandas as pd
    """""Take in the selected session names, and match them to the unique participant-session IDs
    appropriately
    Sample input:
     output_df
       sub01
       sub02
     session_list
       [ses01,ses02]
    Expected output:
     output_df         sessions  participant_sub-1  participant_sub02  participant_sub03
       sub01           ses01                     1                  0                  0 
       sub02           ses01                     0                  1                  0
       sub01           ses02                     1                  0                  0
       sub02           ses02                     0                  1                  0
    """""
#    num_partic_cols = 0
#    for col_names in output_df.columns:
#        if "participant_" in col_names:
#            num_partic_cols += 1
#    if num_partic_cols > 1 and ("Sessions" in output_df.columns or "Sessions_column_one" in output_df.columns):
#        for part_id in pheno_df["participant_id"]:
#            if "participant_{0}".format(part_id) in output_df.columns:
#                continue
#            break
#        else:
#            return output_df
#    return output_df
def main():
    group_config_file = '/home/nrajamani/grp/tests_v1/fsl-feat_config_adhd200_test7.yml'
    output_df_dict= prep_inputs(group_config_file)
    print(output_df_dict)

main()










