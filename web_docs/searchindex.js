Search.setIndex({docnames:["conf","docs_index","index","inet","inet.data","inet.losses","inet.models","inet.models.architectures","inet.models.solvers","inet.models.tf_lite","modules","results","scripts","tests"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["conf.py","docs_index.rst","index.rst","inet.rst","inet.data.rst","inet.losses.rst","inet.models.rst","inet.models.architectures.rst","inet.models.solvers.rst","inet.models.tf_lite.rst","modules.rst","results.rst","scripts.rst","tests.rst"],objects:{"inet.data":[[4,0,0,"-","augmentation"],[4,0,0,"-","constants"],[4,0,0,"-","datasets"],[4,0,0,"-","load_dataset"],[4,0,0,"-","visualization"]],"inet.data.augmentation":[[4,1,1,"","AugmentationMethod"],[4,1,1,"","DataAugmentationHelper"],[4,1,1,"","MultiProbabilityAugmentationMethod"],[4,1,1,"","RandomChannelIntensity"],[4,1,1,"","RandomContrast"],[4,1,1,"","RandomCrop"],[4,1,1,"","RandomFlip"],[4,1,1,"","RandomRotate90"]],"inet.data.augmentation.AugmentationMethod":[[4,2,1,"","label_value_type"],[4,3,1,"","process"]],"inet.data.augmentation.DataAugmentationHelper":[[4,3,1,"","transform"],[4,3,1,"","transform_generator"]],"inet.data.augmentation.MultiProbabilityAugmentationMethod":[[4,4,1,"","method_list"],[4,3,1,"","process"],[4,2,1,"","shared_probabilities"]],"inet.data.augmentation.RandomChannelIntensity":[[4,4,1,"","method_list"],[4,3,1,"","random_scale_n_channels"],[4,3,1,"","random_scale_single_channel"],[4,3,1,"","random_set_single_channel"],[4,2,1,"","shared_probabilities"]],"inet.data.augmentation.RandomContrast":[[4,3,1,"","process"]],"inet.data.augmentation.RandomCrop":[[4,3,1,"","bottom"],[4,3,1,"","bottom_right"],[4,3,1,"","left"],[4,4,1,"","method_list"],[4,3,1,"","right"],[4,2,1,"","shared_probabilities"],[4,3,1,"","top"],[4,3,1,"","top_left"]],"inet.data.augmentation.RandomFlip":[[4,3,1,"","horizontal_flip"],[4,4,1,"","method_list"],[4,2,1,"","shared_probabilities"],[4,3,1,"","vertical_flip"]],"inet.data.augmentation.RandomRotate90":[[4,4,1,"","method_list"],[4,3,1,"","rotate_left"],[4,3,1,"","rotate_right"],[4,2,1,"","shared_probabilities"]],"inet.data.constants":[[4,1,1,"","LabelType"],[4,1,1,"","ModelType"]],"inet.data.constants.LabelType":[[4,2,1,"","MULTI"],[4,2,1,"","NONE"],[4,2,1,"","SINGLE"]],"inet.data.constants.ModelType":[[4,2,1,"","CLASSIFICATION"],[4,2,1,"","REGRESSION"],[4,2,1,"","TWO_IN_ONE"]],"inet.data.datasets":[[4,1,1,"","ImageBoundingBoxDataSet"],[4,1,1,"","ImageDataSet"],[4,1,1,"","ImageLabelDataSet"],[4,1,1,"","ImageTwoInOneDataSet"]],"inet.data.datasets.ImageBoundingBoxDataSet":[[4,2,1,"","label_key"],[4,2,1,"","output_signature"]],"inet.data.datasets.ImageDataSet":[[4,3,1,"","build_dataset"],[4,2,1,"","label_key"],[4,2,1,"","output_signature"]],"inet.data.datasets.ImageLabelDataSet":[[4,2,1,"","label_key"]],"inet.data.load_dataset":[[4,5,1,"","directory_to_classification_dataset"],[4,5,1,"","directory_to_regression_dataset"],[4,5,1,"","directory_to_two_in_one_dataset"]],"inet.data.visualization":[[4,5,1,"","plot_confusion_matrix"],[4,5,1,"","plot_histories"],[4,5,1,"","plot_prediction"],[4,5,1,"","plot_prediction_samples"]],"inet.helpers":[[3,5,1,"","copy_model"],[3,5,1,"","extract_labels_and_features"],[3,5,1,"","get_train_logdir"]],"inet.losses":[[5,0,0,"-","giou_loss"]],"inet.losses.giou_loss":[[5,1,1,"","GIoULoss"],[5,1,1,"","LossFunctionWrapper"],[5,5,1,"","convert_values"],[5,5,1,"","giou_loss"],[5,5,1,"","tf_giou_loss"]],"inet.losses.giou_loss.LossFunctionWrapper":[[5,3,1,"","call"],[5,3,1,"","get_config"]],"inet.models":[[6,0,0,"-","data_structures"],[6,0,0,"-","hyper_parameter_optimization"]],"inet.models.architectures":[[7,0,0,"-","base_model"],[7,0,0,"-","bounding_boxes"],[7,0,0,"-","classifier"]],"inet.models.architectures.base_model":[[7,1,1,"","Backbone"],[7,1,1,"","Model"],[7,1,1,"","SingleTaskModel"],[7,1,1,"","TaskModel"]],"inet.models.architectures.base_model.SingleTaskModel":[[7,3,1,"","evaluate_predictions"],[7,3,1,"","from_config"]],"inet.models.architectures.base_model.TaskModel":[[7,3,1,"","default_callbacks"],[7,3,1,"","evaluate_model"],[7,3,1,"","evaluate_predictions"],[7,3,1,"","extract_backbone_features"],[7,3,1,"","fit"],[7,2,1,"","model_type"],[7,3,1,"","to_tflite"]],"inet.models.architectures.bounding_boxes":[[7,1,1,"","BoundingBoxHyperModel"],[7,1,1,"","BoundingBoxRegressor"]],"inet.models.architectures.bounding_boxes.BoundingBoxHyperModel":[[7,3,1,"","build"],[7,2,1,"","model_data"]],"inet.models.architectures.bounding_boxes.BoundingBoxRegressor":[[7,3,1,"","compile"],[7,3,1,"","evaluate_predictions"],[7,2,1,"","model_type"]],"inet.models.architectures.classifier":[[7,1,1,"","Classifier"],[7,1,1,"","ClassifierHyperModel"]],"inet.models.architectures.classifier.Classifier":[[7,3,1,"","compile"],[7,3,1,"","evaluate_predictions"],[7,2,1,"","model_type"]],"inet.models.architectures.classifier.ClassifierHyperModel":[[7,3,1,"","build"],[7,2,1,"","model_data"],[7,2,1,"","weights"]],"inet.models.data_structures":[[6,1,1,"","BoundingBox"],[6,1,1,"","ModelArchitecture"]],"inet.models.data_structures.BoundingBox":[[6,3,1,"","A_I"],[6,3,1,"","A_U"],[6,3,1,"","GIoU"],[6,3,1,"","IoU"],[6,4,1,"","area"],[6,3,1,"","draw"],[6,4,1,"","half_h"],[6,4,1,"","half_w"],[6,3,1,"","overlap"],[6,4,1,"","x_max"],[6,4,1,"","y_max"]],"inet.models.data_structures.ModelArchitecture":[[6,2,1,"","backbone"],[6,2,1,"","create_model"],[6,2,1,"","name"]],"inet.models.hyper_parameter_optimization":[[6,1,1,"","FrozenBlockConf"],[6,5,1,"","plot_hpo_values"],[6,5,1,"","read_trials"]],"inet.models.hyper_parameter_optimization.FrozenBlockConf":[[6,2,1,"","TRAIN_ALL"],[6,2,1,"","TRAIN_HALF"],[6,2,1,"","TRAIN_NONE"],[6,3,1,"","choices"],[6,3,1,"","process"]],"inet.models.solvers":[[8,0,0,"-","common"],[8,0,0,"-","independent"],[8,0,0,"-","tf_lite"],[8,0,0,"-","two_in_one"],[8,0,0,"-","two_stage"]],"inet.models.solvers.common":[[8,5,1,"","evaluate_solver_predictions"]],"inet.models.solvers.independent":[[8,1,1,"","IndependentModel"]],"inet.models.solvers.independent.IndependentModel":[[8,2,1,"","model_name"],[8,3,1,"","predict"]],"inet.models.solvers.tf_lite":[[8,1,1,"","MultiTaskModel"]],"inet.models.solvers.tf_lite.MultiTaskModel":[[8,3,1,"","create_classifier"],[8,3,1,"","create_regressor"],[8,3,1,"","crop_image"],[8,3,1,"","evaluate_model"],[8,3,1,"","from_config"],[8,2,1,"","model_name"],[8,3,1,"","predict"]],"inet.models.solvers.two_in_one":[[8,1,1,"","TwoInOneHyperModel"],[8,1,1,"","TwoInOneModel"],[8,1,1,"","TwoInOneTFLite"]],"inet.models.solvers.two_in_one.TwoInOneHyperModel":[[8,3,1,"","build"],[8,2,1,"","model_data"]],"inet.models.solvers.two_in_one.TwoInOneModel":[[8,3,1,"","compile"],[8,3,1,"","evaluate_model"],[8,3,1,"","evaluate_predictions"],[8,3,1,"","from_config"],[8,2,1,"","model_type"]],"inet.models.solvers.two_in_one.TwoInOneTFLite":[[8,3,1,"","evaluate_model"],[8,3,1,"","from_config"],[8,3,1,"","predict"]],"inet.models.solvers.two_stage":[[8,1,1,"","TwoStageModel"]],"inet.models.solvers.two_stage.TwoStageModel":[[8,2,1,"","model_name"],[8,3,1,"","predict"]],"inet.models.tf_lite":[[9,0,0,"-","convert_to_tflite"],[9,0,0,"-","tflite_methods"]],"inet.models.tf_lite.convert_to_tflite":[[9,1,1,"","ClusterMethod"],[9,1,1,"","QuantizationMethod"],[9,5,1,"","cluster_weights"],[9,5,1,"","create_pruned_model"],[9,5,1,"","create_q_aware_model"],[9,5,1,"","create_quantize_model"],[9,5,1,"","create_tf_lite_q_model"]],"inet.models.tf_lite.convert_to_tflite.ClusterMethod":[[9,2,1,"","DENSITY_BASED"],[9,2,1,"","KMEANS_PLUS_PLUS"],[9,2,1,"","LINEAR"],[9,2,1,"","RANDOM"]],"inet.models.tf_lite.convert_to_tflite.QuantizationMethod":[[9,2,1,"","DYNAMIC"],[9,2,1,"","FLOAT_16"],[9,2,1,"","FULL_INT"],[9,2,1,"","NONE"]],"inet.models.tf_lite.tflite_methods":[[9,5,1,"","evaluate_classification"],[9,5,1,"","evaluate_interpreted_model"],[9,5,1,"","evaluate_q_model"],[9,5,1,"","evaluate_regression"],[9,5,1,"","evaluate_two_in_one"],[9,5,1,"","get_gzipped_model_size"],[9,5,1,"","save_model_file"],[9,5,1,"","validate_q_model_prediction"]],"scripts.configs":[[12,5,1,"","create_config"],[12,5,1,"","create_conversion_config"],[12,5,1,"","create_tflite_config"]],"scripts.helpers":[[12,1,1,"","ProgressBar"],[12,1,1,"","ThreadWithReturnValue"],[12,5,1,"","decision"],[12,5,1,"","move_files"]],"scripts.helpers.ProgressBar":[[12,3,1,"","done"],[12,3,1,"","step"]],"scripts.helpers.ThreadWithReturnValue":[[12,3,1,"","join"],[12,3,1,"","run"]],"scripts.preselect_files":[[12,5,1,"","calculate_stats"],[12,5,1,"","create_local_copy"],[12,5,1,"","decide_image"],[12,5,1,"","get_directory_by_prefix"],[12,5,1,"","get_id_range_for_search_term"],[12,5,1,"","get_id_ranges_from_input_directory"],[12,5,1,"","get_n_random_elements"],[12,5,1,"","init_random_dataset_ids"],[12,5,1,"","process_directories"],[12,5,1,"","process_directory"],[12,5,1,"","scan_input_directory"],[12,5,1,"","test_dataset"]],"scripts.process_files":[[12,5,1,"","bounding_box_stats"],[12,5,1,"","create_config_file"],[12,5,1,"","create_dataset_structure"],[12,5,1,"","create_directory_structure"],[12,5,1,"","extract_file_name"],[12,5,1,"","extract_label"],[12,5,1,"","generate_statistics"],[12,5,1,"","get_genera_file_stats_for_directory"],[12,5,1,"","load_element"],[12,5,1,"","load_labels_from_bbox_file"],[12,5,1,"","load_labels_from_bbox_files"],[12,5,1,"","split_labeled_files"],[12,5,1,"","spread_files"],[12,5,1,"","test_output_directory"]],"scripts.reuse_labels":[[12,5,1,"","extract_file_name"],[12,5,1,"","get_directory_from_prefix"],[12,5,1,"","move_file"],[12,5,1,"","process_in_multi_threads"],[12,5,1,"","process_in_single_thread"]],"tests.helper":[[13,1,1,"","Timer"],[13,5,1,"","build_tf_model_from_file"],[13,5,1,"","evaluate_q_model_from_file"]],"tests.test_yolo_inference":[[13,5,1,"","filter_classes"],[13,5,1,"","process_best_prediction"],[13,5,1,"","scale_bb"],[13,5,1,"","yolo2voc"]],inet:[[3,0,0,"-","helpers"]],scripts:[[12,0,0,"-","configs"],[12,0,0,"-","constants"],[12,0,0,"-","convert_models_to_tflite"],[12,0,0,"-","generate_cropped_dataset"],[12,0,0,"-","helpers"],[12,0,0,"-","preselect_files"],[12,0,0,"-","process_files"],[12,0,0,"-","reuse_labels"]],tests:[[13,0,0,"-","helper"],[13,0,0,"-","test_tf_architectures"],[13,0,0,"-","test_tf_lite_architectures"],[13,0,0,"-","test_yolo_inference"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","method","Python method"],"4":["py","property","Python property"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:method","4":"py:property","5":"py:function"},terms:{"0":[0,1,3,4,6,7,8,11,12,13],"001":8,"00420":12,"0061":11,"006126":11,"02":1,"026241":11,"0525":11,"0590909090909091":11,"0591":11,"06":7,"1":[1,3,4,6,7,8,9,11,12,13],"10":12,"100":[4,11,12],"1024":4,"10px":12,"12":[1,7,8],"125":7,"128":7,"13":3,"14":1,"15":9,"17":11,"1733":11,"17333333333333334":11,"18":11,"19":11,"1935":11,"193527":11,"1e":[7,8,9],"2":[1,4,5,6,7,8,9],"20":[1,7,11,12],"200px":12,"2022":[0,1],"20px":12,"21":[3,11],"2111":11,"211178":11,"2112":11,"2184":11,"218405":11,"224":[4,7,8],"24":11,"2404":11,"25":[1,4,7,8,11],"255":4,"26":11,"3":[1,4,7,8,9,11],"30":12,"300px":12,"3042":11,"32":[4,7],"324131":11,"33":1,"3333333333333333":4,"344194":11,"3442":11,"34496":11,"347866":11,"3566":11,"36":[7,8],"3604":11,"3670":11,"372694":11,"3727":11,"38":11,"3838":11,"3877":11,"39":11,"399976":11,"4":[4,5,7,8],"40":11,"4000":11,"42":[1,4,7,8],"43":11,"4359699999999997":11,"4361618":1,"43616188":11,"43982178":11,"44598112":11,"4623789":11,"4624":11,"4625":11,"4625029774807487":11,"467743":11,"4751":11,"4751697":11,"4762":11,"4762167417804296":11,"4844":11,"48444444444444446":11,"4933":11,"49333333333333335":11,"5":[4,7,8,11],"50":[3,4,7,8],"5097":11,"5097587081088926":11,"518587":11,"5186":11,"52":11,"5269":11,"5284798070186199":11,"5288888888888889":11,"5289":11,"544":11,"5522557272067077":11,"5602":11,"5661":11,"5666":11,"5730777":11,"5731":11,"5803820077663839":11,"5e":[7,8],"6":4,"608":11,"61227506":11,"6161582":11,"62672853":11,"6298":11,"62984677235293":11,"6330171":11,"6353292590535011":11,"63962024":11,"64":7,"6422":11,"6422222222222222":11,"64344555":11,"672493":11,"680496":11,"6805":11,"6894023":12,"6904543":12,"6958105":11,"7":4,"7224772507688828":11,"7296":11,"729637":11,"7311":11,"7311111111111112":11,"731425":11,"7438":11,"743893":11,"7496":11,"7496034996537605":11,"75":8,"7555":11,"7555555555555555":11,"7779":11,"77794664790425":11,"7793":11,"779331433620758":11,"78":11,"7800":11,"7844":11,"7844444444444445":11,"786606":11,"8":1,"8080":1,"81673026":11,"8261":11,"8261255977635148":11,"8333":11,"8333333333333334":11,"8367":11,"8367397903530476":11,"84":11,"8438866903702877":11,"8439":11,"8462":11,"8462153544896847":11,"8466666666666667":11,"8467":11,"8511":11,"8511111111111112":11,"9":12,"909544":11,"916":[1,11],"9167668857681328":[1,11],"917557":11,"9176":11,"9196824463343246":11,"92":11,"9209747544826291":11,"947507":11,"abstract":4,"boolean":7,"case":[1,4,9],"class":[4,5,6,7,8,9,12,13],"default":[0,1,5,7,8],"do":11,"enum":[4,6,9],"export":1,"final":12,"float":[4,5,6,7,8,12],"function":[3,4,5,7,8,12],"import":[0,7,8,12,13],"int":[3,4,6,7,8,12],"new":[3,6,7,8,12],"return":[3,4,5,6,7,8,9,12,13],"static":[0,4,6,7,8,12],"true":[0,1,4,5,7,8,12],"while":7,A:[1,2,5,7,8],For:[0,1,7,12],If:0,In:[1,8,9],No:4,One:8,The:[0,1,5,11,13],To:[1,11,12],__file__:0,__perform_transform:4,_static:0,_templat:0,a_i:6,a_u:6,abc:4,absolut:0,abspath:0,acc:11,access:1,accommod:1,accord:[1,4],accordingli:[1,4],account:1,accuraci:[1,4,7,8],activ:7,ad:12,adam:[7,8],add:[0,7,8],add_module_nam:0,addit:[7,8],addon:5,affect:0,afmhot_r:4,after:[0,4],afterward:8,alia:7,all:[1,12],allow:[4,12],alpha:[7,8],also:0,amount:12,an:[1,4,5,6,7,8,9],ani:[0,4,12],annot:[1,12],anoth:[0,6,12],app:1,append:7,appli:[4,7,8,9,11],applic:[7,8],approv:12,ar:[0,3,5],arch:11,architectur:[3,6,8,9,10],area:[4,6],arg:[5,7,8,12],around:[1,4,6],arrai:4,ask:12,attribut:4,augment:[3,10],augmentationmethod:4,augmented_dataset:4,author:0,auto:5,autodoc:0,autosummary_gener:0,autumn_r:4,avail:[4,6,12],avg:[11,12],awar:9,axi:4,b2dafcfa74c5de268b8a5c53813bc0b89cadf386:5,b:0,backbon:[6,7,8],bar:12,base:[3,4,5,6,7,8,9,12,13],base_dir:12,base_model:[3,6,8,9,10],batch:7,batch_siz:[4,7],bayesianoptim:[7,8],bb1:5,bb2:[5,6],bb:[4,6,8,13],bbox:[6,7,8,12,13],bbox_fil:12,bbox_label_index:4,bbreg:[8,9,11],been:11,befor:[7,8,12],behavior:4,best:13,between:4,binari:4,blob:5,block:[1,7],blue:4,bone_r:4,bool:[7,8,12],bootstrap:8,border:4,both:[6,8],bottom:4,bottom_right:4,bound:[1,4,5,7,8,12,13],bounding_box:[3,6,8,10],bounding_box_stat:12,boundingbox:6,boundingboxhypermodel:7,boundingboxlabeltyp:4,boundingboxregressor:[7,8],box:[1,4,5,7,8,12,13],brg_r:4,bugn:4,build:[4,7,8,12],build_dataset:4,build_tf_model_from_fil:13,builder:0,builtin:0,bupu:4,bwr_r:4,by_nam:8,bypass:5,c:8,calcul:[5,7,8,12],calculate_stat:12,call:[5,9,12],callabl:[4,6,7,8],callback:7,can:[0,1,4,11,12],categorical_crossentropi:7,centroidiniti:9,cfg:[7,8],chang:[4,9,12],channel:4,child:4,choic:6,chose:4,chosen:4,cividis_r:4,class_nam:[4,12],classes_in:13,classif:[2,4,7,8,9],classifi:[3,6,8,10,11],classifierhypermodel:7,classlabeltyp:4,classmethod:[7,8],clf:7,clf_backbon:8,cluster:9,cluster_method:9,cluster_weight:9,clustermethod:9,cnn:8,code:1,collect:[7,12],color:[4,6],colormap:4,com:[5,12],combin:[4,9],combined_list:12,come:0,common:[0,3,6,10],commonli:12,compar:9,comparison:2,compat:[1,7],compil:[7,8],comput:[5,6,7,8,9,11],confid:4,config:[8,10],configur:[0,4,5,6,7,8,12],confus:[4,7,8],consist:[7,12],constant:[3,7,8,10],constructor:[4,8],consult:[1,12],contain:[0,1,4,6,8,12],content:1,context:6,contrast:4,convers:12,convert:[4,5,7,9,12,13],convert_models_to_tflit:10,convert_to_tflit:[3,6,10],convert_valu:5,convex:6,cool_r:4,coordin:[5,6,12],copi:[0,3,12],copy_model:3,copyright:0,correct:12,count:12,cours:[4,6],creat:[1,7,8,9,12,13],create_classifi:8,create_config:12,create_config_fil:12,create_conversion_config:12,create_dataset_structur:12,create_directory_structur:12,create_local_copi:12,create_model:6,create_pruned_model:9,create_q_aware_model:9,create_quantize_model:9,create_regressor:8,create_tf_lite_q_model:9,create_tflite_config:12,creation:6,crop:[1,4,8,12],crop_imag:8,css:0,current:[3,7,8,12],custom:[0,12],cut:4,cycl:4,data:[3,5,7,8,10,12],data_structur:[3,7,8,10],dataaugmentationhelp:4,dataclass:6,dataset:[3,7,10,12],dataset_op:4,datasetv2:4,decid:[4,5],decide_imag:12,decis:12,decreas:4,dedic:[7,12],default_callback:7,defin:4,definit:8,demo:1,dens:7,dense_neuron:7,density_bas:9,depend:[8,12],descript:2,detail:1,detect:[1,5,8],determin:[4,9],develop:[1,6],dict:[8,12],dictionari:[8,12],differ:[1,6,7],dir_nam:[6,12],dir_prefix:12,directori:[0,1,3,4,7,8,12],directory_to_classification_dataset:4,directory_to_regression_dataset:4,directory_to_two_in_one_dataset:4,dirnam:0,displai:6,doc:[0,1],docker:1,document:[0,1],domain:9,done:[4,7,8,9,12],download:2,draw:6,dropout:[7,8,11],dropout_factor:7,dtype:4,due:11,dure:[4,6,7,8,11],dynam:9,e:[1,6,12],each:[4,5,12,13],earlystop:7,edu:[5,6],either:5,elem:[8,12],element:12,en:0,encod:[5,13],engin:[6,7,8],enhanc:5,entri:11,enumer:4,environ:[1,4],epoch:[7,8],essenti:[4,7,8],evalu:[6,7,8,9,11,13],evaluate_classif:9,evaluate_interpreted_model:9,evaluate_model:[7,8],evaluate_predict:[7,8],evaluate_q_model:9,evaluate_q_model_from_fil:13,evaluate_regress:9,evaluate_solver_predict:8,evaluate_two_in_on:9,exampl:[3,4,7,8,12,13],except:11,exclude_pattern:0,execut:[1,12],expect:12,expected_num_el:12,ext:0,extend:[7,8],extens:0,extract:[3,4,8,12],extract_backbone_featur:7,extract_file_nam:12,extract_label:12,extract_labels_and_featur:3,extractor:7,f1:[1,7,8],f:[7,8],factor:[4,7,8],fals:[0,4,7,8,12],featur:[3,4,7,8,12],feed:7,file:[0,1,6,8,9,12,13],file_cont:12,file_count:12,file_list:12,filenam:13,filter:12,filter_class:13,find:[11,12],first:[1,4,5],fit:7,flag:7,flip:4,float32:4,float_16:9,fn:[5,11],foo:3,fork:5,form:4,format:13,found:[1,12],fp:11,framework:[4,5],freez:6,from:[1,3,4,7,8,12,13],from_config:[7,8],frozen:7,frozen_lay:7,frozenblockconf:[6,7],full:[0,4],full_int:9,further:7,g:[1,6,12],gc:6,gca:6,gener:[0,3,4,5,6,12],genera:[1,12],generate_cropped_dataset:[1,10],generate_statist:12,genu:12,get:[4,6,7,8,12],get_config:5,get_directory_by_prefix:12,get_directory_from_prefix:12,get_genera_file_stats_for_directori:12,get_gzipped_model_s:9,get_id_range_for_search_term:12,get_id_ranges_from_input_directori:12,get_n_random_el:12,get_train_logdir:3,getter:[5,12],giou:[1,5,6,7,8],giou_loss:[3,10],giouloss:[5,11],github:5,given:[3,8,9,12],global:7,gnbu:4,gnu:1,good:4,gpl:1,graphic:6,green:4,grei:4,ground:[4,5,7,8,9],group:12,gzip:9,h5:[7,8,9],h:[1,5,6,12,13],half:6,half_h:6,half_w:6,harmon:11,have:[3,11],heartexlab:1,height:[4,6,7,12,13],help:[0,1],helper:[4,6,7,8,9,10],here:[0,8],hidden:7,hint:9,hist:4,hold:[1,4,8,12,13],hood:4,horizont:4,horizontal_flip:4,hot:13,hp:[6,7,8],hpo:[6,7,8],hpo_model:[7,8],html:0,html_extra_path:0,html_static_path:0,html_theme:0,http:[0,5,6,12],hull:6,human:7,hyper:[7,8],hyper_parameter_optim:[3,10],hypermodel:[7,8],i:12,id:12,ignor:0,imag:[2,4,8,9,11,12,13],imageboundingboxdataset:4,imagedataset:[4,7],imagelabeldataset:4,imagenet:[7,8],imagetwoinonedataset:4,img_height:4,img_path:12,img_width:4,implement:[4,5,7,8],in_dir:12,in_directori:12,inat:[1,12],inaturalist:12,include_scor:4,include_top:[7,8],increas:4,independ:[3,6,10,11,12],independentmodel:[1,8],index:[2,12],indic:13,individu:[1,4,7,8],inet:0,infer:[2,13],inform:0,inherit:4,init_random_dataset_id:12,input:[3,4,7,8,9,12,13],input_dir:12,input_directori:1,input_model:3,input_shap:[7,8,12],insect:2,insert:0,insid:4,instal:2,instanc:[1,3,4,5,6,8,9],integr:12,intens:4,interfac:8,intern:5,interpret:[9,13],intersect:[5,6],intersphinx:0,introduc:5,invok:5,io:6,iou:[5,6],is_tflit:8,issu:[1,5],iter:[4,7,8,12,13],its:[4,9],join:[0,12],json:[1,8,12],just:4,kei:4,kera:[5,6,7,8],keras_tun:[6,7,8],kernel:7,kinsektdaten:1,kmeans_plus_plu:9,kt:[7,8],kwarg:[5,7,8,12],l2:7,l:1,label:[3,4,7,8,9,12,13],label_kei:4,label_nam:12,label_studio_local_files_document_root:1,label_studio_local_files_serving_en:1,label_value_typ:4,labelstudio:1,labeltyp:[4,7],latest:1,launch:1,layer:[3,6,7],layer_rang:3,learn:[2,7,8],learning_r:[7,8],left:4,licens:2,like:0,linear:9,linearsegmentedcolormap:4,list:[0,4,6,7,8,12],lite:9,load:[7,8,12],load_dataset:[3,10],load_el:12,load_labels_from_bbox_fil:12,load_weight:[7,8],local:[2,8,12],locat:[1,8,12],log:11,look:[0,1,2,12],loss:[3,4,7,8,10,11],loss_weight:8,lossfunctionwrapp:5,losswrapp:5,lstudio:1,m2r2:0,m:[1,12],machin:2,maintain:12,make:[0,7],map:4,mark:1,markdown:0,master:0,match:[0,12],mathjax:0,matplotlib:[4,6],matrix:[4,7,8],max:[4,6,7,8,12],max_num_el:12,max_trial:[7,8],md:0,mean:[8,11,12],measur:13,messag:12,method:[2,3,4,6,7,8,9,11,12,13],method_list:4,metric:[2,5,7,8],micro:11,min:[7,12],ml:1,mnt:1,mobilenet:[7,8,9],mode:5,model:[1,2,3,4,5,10,12,13],model_data:[7,8],model_nam:[7,8,9],model_predict:9,model_typ:[7,8,9],modelarchitectur:[6,7,8],modelcheckpoint:7,modeltyp:[4,7,8,9],modul:[0,2,10],monitor:7,monitoring_v:[7,8],more:1,most:[0,3],mostli:6,mount:12,mount_directori:1,move:12,move_fil:12,move_files_onli:12,mse:7,multi:[4,12],multipl:[4,12],multipli:4,multiprobabilityaugmentationmethod:4,multitask:8,multitaskmodel:8,my:[7,8],my_dataset:4,my_model:7,my_weight:7,n:4,na:1,name:[0,3,4,5,6,7,8,9,12],need:1,neg:4,nest:12,neuron:7,next:[4,7,8],none:[3,4,5,6,7,8,9,12],normal:[4,7,13],note:[4,9],notebook:6,noth:8,num_class:7,num_el:12,num_initial_point:[7,8],num_sampl:12,number:[7,12],number_clust:9,number_sampl:4,object:[1,4,5,6,7,8,12,13],off:4,one:[4,5,9,12,13],ones:0,onli:0,onto:[4,9],op:[4,5],openfortivpn:1,oper:4,optim:[1,7,8,11],option:[0,1,3,4,5,6,7,8],orang:4,order:12,org:0,origin:[4,8,9,13],orrd:4,os:0,other:6,out:[8,12],out_dir:12,output:[0,7,8,12,13],output_directori:1,output_fil:12,output_signatur:4,over:[4,5,6,12],overal:8,overlap:6,overrid:4,overwrit:[0,7,8],overwritten:12,p:1,packag:[2,10],page:[0,1,2],pair:4,paper:[1,2,6,11],param:[4,6,12],paramet:[3,4,5,7,8,9,12,13],parent:[7,8,12],parent_directori:4,pars:6,part:[1,4],pass:[4,7,8],password:1,patch:8,path:[0,12],pattern:0,pb:12,pct:[4,12],pdf:[1,5,6],per:[5,12],percent:4,percentag:[4,13],perform:[4,7,8,9,13],phase:4,philipp:0,pipelin:4,pixel:[4,12],pleas:1,plot:[4,8],plot_confusion_matrix:4,plot_histori:4,plot_hpo_valu:6,plot_predict:4,plot_prediction_sampl:4,pool:7,popul:3,posit:4,potenti:4,power:[7,8],precis:11,predecessor:9,predict:[4,5,7,8,9,13],predicted_bb:[4,8],predicted_label:[4,8],prefix:[3,12],prepar:12,preprocess:8,preprocessing_method:[7,8],preselect_fil:[1,10],preserv:9,pretrain:[1,13],previou:12,print:[4,12,13],prior:[4,9],probabl:4,process:[4,6,7,12],process_best_predict:13,process_directori:12,process_fil:[1,10],process_in_multi_thread:12,process_in_single_thread:12,progbarlogg:7,progress:12,progressbar:12,proj_nam:[7,8],project:[0,2],project_nam:[7,8],propag:6,properti:[4,6],provid:[4,6,8,9,12],prune:9,publish:1,pubu:4,pubugn:4,purd:4,purpl:4,pwd:1,py:[0,4,5,12],pyplot:[4,6],python:[1,4,5,6,7,12],q:9,q_model:9,quant_method:9,quantiz:[9,11],quantization_method:7,quantizationmethod:[7,9],random:[4,9,12],random_scale_n_channel:4,random_scale_single_channel:4,random_set_single_channel:4,randomchannelintens:4,randomcontrast:4,randomcrop:4,randomflip:4,randomli:4,randomrotate90:4,rang:[3,12],raspberrypi:2,rate:[7,8],raw:8,rdpu:4,read:6,read_trial:6,readabl:7,real:13,reappli:12,recal:11,receiv:8,record:12,red:[4,6],reducelronplateau:7,reduct:5,reg:7,reg_backbon:8,regress:[1,4,5,7,8,9],regressor:[7,8],regular:[7,8,9,11],regularization_factor:7,rel:0,relat:4,releas:1,relu:7,render:[4,6,7,8],render_sampl:[7,8],renku:0,repositori:1,repres:9,represent:[6,9],requir:[1,3,4,12],requires_decis:12,research:1,restructuredtext:0,result:[2,4,8,9,13],retrain:11,reuse_label:[1,10],review:11,right:4,rmse:[7,8],rng:1,root:[0,12],root_directori:12,rotat:4,rotate_left:4,rotate_right:4,rst:0,run:[1,11,12],run_2022_06_09:3,s:[1,11],same:[2,4],sampl:[4,5,7,8,9,11,12,13],save:8,save_model_fil:9,sc:0,scale:[4,13],scale_bb:13,scan_input_directori:12,score:[1,7,8,11],script:[0,1,2,10],search:[2,7,8,12],search_term:12,see:[0,1,4,7],seed:[1,4,7,8],select:[0,6,7,8,13],self:6,send:12,sequenti:[4,6,8],set:[3,4,7,8,9,12],set_nam:4,setup:0,sh:1,shadow:8,shape:[4,5,12],share:12,shared_prob:4,sheet:0,shell:1,shown:0,shuffl:4,side:4,similar:8,simplifi:6,simultan:[1,8],singl:[1,2,4,7,8,12],singlestagemodel:1,singletaskmodel:7,size:[7,9,12,13],sleep:[12,13],so:[0,11],softmax:7,solv:[1,2,7,8],solver:[3,6,10],some_imag:7,some_input:[7,8],sourc:[0,1,3,4,5,6,7,8,9,12,13],source_suffix:0,speci:12,sphinx:0,split:12,split_labeled_fil:12,spread_fil:12,src:1,stackoverflow:12,stage:[8,11,12],stanford:[5,6],state:[7,8],statist:12,statu:2,stdout:12,step:12,storag:[1,7,12],storage_dir:12,store:[7,12],str:[3,4,5,6,7,8,12],string:[0,4],structur:[1,12],studio:1,style:0,sub:3,subdirectori:12,submit:1,submodul:10,subpackag:10,sudo:1,support:2,sy:0,tailor:7,target:[1,5,12],target_dir:12,target_directori:12,task:[1,2,7,8,9],taskmodel:[6,7,8,9],techniqu:[4,11],templat:0,templates_path:0,tensor:5,tensor_spec:4,tensorboard:7,tensorflow:[4,5,6,7,8],tensorflow_addon:5,tensorspec:4,term:12,terminateonnan:7,test:[0,2,9,10,12],test_dataset:12,test_imag:[9,13],test_label:9,test_output_directori:12,test_set:[7,9,12],test_shar:12,test_split:12,test_tf_architectur:10,test_tf_lite_architectur:10,test_yolo_infer:10,tf:[4,5,9,12],tf_giou_loss:5,tf_lite:[3,6,10],tf_lite_model:[9,13],tfl_model_predict:9,tflite:[1,7,8,9,12,13],tflite_method:[3,6,10],thei:0,theme:0,theoret:1,thesi:2,thesisphilipp:1,thi:[0,1,4,7,9,12],thing:1,those:11,thread:12,threadwithreturnvalu:12,three:12,through:[6,8],ticket:1,time:[2,3,12,13],timer:13,titl:4,tn:11,to_tflit:7,top:4,top_left:4,tp:11,tracker:1,train:[4,7,8,9,11,12,13],train_al:[6,7],train_half:[6,7],train_insecta:1,train_log:3,train_non:[6,7],train_set:[7,8,9,12],transform:[4,13],transform_gener:4,treat:8,trial:6,true_bb:4,truth:[4,5,7,8,9],tunabl:[7,8],tuner:[6,7,8],tupl:[3,4,7,8,12],two:[2,8,9,12],two_in_on:[3,4,6,10],two_stag:[3,6,10],twoinon:8,twoinonehypermodel:8,twoinonemodel:8,twoinonetflit:8,twostagemodel:[1,8],txt:0,unavail:5,unbatch:[7,8],under:[1,4],underli:[1,9,12],union:[4,5,6,12],unless:1,up:[1,4,7,8],upload:1,us:[0,4,5,6,7,8,9,11,12,13],usag:[0,2,12],usernam:1,v1:1,v:1,val:1,val_accuraci:[7,8],val_loss:7,val_set:12,val_split:12,valid:[1,7,8,9,12],validate_q_model_predict:9,validation_bb:4,validation_featur:4,validation_label:8,validation_set:[7,8],validation_shar:12,validation_valu:8,valu:[4,5,6,7,9,11,12],variabl:1,varianc:11,vector:[4,8],verbatim:4,verbos:[7,12],verifi:12,version:[1,5,9,12],vertic:4,vertical_flip:4,vgg16:7,vgg_backbon:7,via:12,viewcod:0,virtualenv:1,visual:[2,3,7,10],voc:13,volum:12,vpn:1,vpn_passwd:1,vpn_user:1,w:[5,6,13],wa:5,want:9,webapp:1,weight:[7,8,9,13],well:[1,4,9],when:[0,6,7,8,9],which:5,width:[4,6,7,12,13],wistia:4,within:[1,12],without:11,wrap:[4,5],wrapper:[4,5,7,8],write:9,www:0,x1:13,x2:13,x:[5,6,7,8,11,12],x_hat:8,x_max:[5,6,12],x_min:[5,12],xmid:13,y1:13,y:[4,5,6,7,8],y_max:[5,6,12],y_min:[5,12],y_pred:[4,5],y_true:[4,5],yield:4,ylgn:4,ylgnbu:4,ylorbr:4,ylorrd:4,ymid:13,yolo2voc:13,yolo:13,yolov5:[1,13],you:[1,9,11],your:[0,9],zettl:0},titles:["&lt;no title&gt;","iNet","iNet","inet.package","inet.data package","inet.losses package","inet.models package","inet.models.architectures package","inet.models.solvers package","inet.models.tf_lite package","inet","Results","scripts package","tests package"],titleterms:{"16":11,"2":11,"2head":11,"2head1":11,"2head2":11,"default":11,accuraci:11,architectur:7,augment:[1,4,11],base_model:7,best:11,bounding_box:7,classif:[1,11],classifi:7,clf1:11,clf2:11,clf3:11,clf:11,common:8,comparison:11,config:12,constant:[4,12],content:2,convert_models_to_tflit:12,convert_to_tflit:9,data:[1,4],data_structur:6,dataset:[1,4,11],descript:1,f1:11,gener:1,generate_cropped_dataset:12,giou:11,giou_loss:5,head:11,helper:[3,12,13],hyper_parameter_optim:6,imag:1,independ:8,indic:2,inet:[1,2,3,4,5,6,7,8,9,10,11],infer:[1,11],insect:1,instal:1,label:1,learn:1,licens:1,load_dataset:4,local:[1,11],loss:5,machin:1,method:1,metric:11,mobilenet:11,model:[6,7,8,9,11],modul:[3,4,5,6,7,8,9,12,13],origin:11,packag:[3,4,5,6,7,8,9,12,13],pre:1,predict:1,predictor:11,prerequesit:1,preselect_fil:12,process_fil:12,project:1,py:1,raspberrypi:11,recreat:1,reg1:11,reg2:11,reg:11,regress:11,result:11,reuse_label:12,rmse:11,same:11,script:12,set:1,singl:11,solv:11,solver:8,statu:1,submodul:[3,4,5,6,7,8,9,12,13],subpackag:[3,6],support:1,tabl:2,task:11,test:[1,11,13],test_tf_architectur:[1,13],test_tf_lite_architectur:[1,13],test_yolo_infer:[1,13],tf_lite:[8,9],tflite:11,tflite_method:9,thesi:1,time:11,train:1,two:11,two_in_on:8,two_stag:8,uncrop:11,usag:1,vgg:11,visual:[1,4]}})