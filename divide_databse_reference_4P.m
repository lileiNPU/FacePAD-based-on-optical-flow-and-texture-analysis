clc;
clear;
close all;

% image path
% save path
save_path_train = '.\data\DataList4P';
if ~exist(save_path_train)
    mkdir(save_path_train);
end

% length of video sequence
num_frame = 75;
over_lap_frame =  10*3;
face_mode = 'AllFrameImages4P';

%% train set
% real face
set_img_path = {['../', face_mode]};
client_id = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        img_name = img_files(i).name;
        img_id = img_name;
        index_cut = strfind(img_id, '_');
        client_id{end + 1} = img_id(1 : index_cut(end));   
    end
end
client_id = unique(client_id);
train_id_all = client_id;

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    img_name = img_files(i).name;
    img_id = img_name;
    index_cut = strfind(img_id, '_');
    id_all{end + 1} = img_id(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(train_id_all)
    id_sub = train_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    for index = 1 : over_lap_frame : img_num
        index_start = index;
        index_end = index + num_frame - 1;
        if index_end > img_num
            continue;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_train, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for img_index = index_start : index_end
            img_name = strcat(id_sub{1}, num2str(img_index), '.jpg');
            if ~isempty(strfind(id_sub{1}, '_1_counter_'))  
                img_label = 0;
                img_index = 0;
            else
                img_label = 1;
                img_index = 1;
            end
            
            fprintf(fid,'%s %s %s \n', fullfile(set_img_path{1}, img_name), num2str(img_label), num2str(img_index));  
        end
        fclose(fid);
    end
end
