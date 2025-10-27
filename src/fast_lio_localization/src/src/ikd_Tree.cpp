#include "ikd_Tree.h"

/*
Description: ikd-Tree: an incremental k-d tree for robotic applications 
Author: Yixi Cai
email: yixicai@connect.hku.hk
*/

template <typename PointType>
KD_TREE<PointType>::KD_TREE(float delete_param, float balance_param, float box_length) {
    delete_criterion_param = delete_param;
    balance_criterion_param = balance_param;
    downsample_size = box_length;
    Rebuild_Logger.clear();           
    termination_flag = false;
    start_thread();
}

template <typename PointType>
KD_TREE<PointType>::~KD_TREE()
{
    stop_thread();
    Delete_Storage_Disabled = true;
    delete_tree_nodes(&Root_Node);
    PointVector ().swap(PCL_Storage);
    Rebuild_Logger.clear();           
}

template <typename PointType>
void KD_TREE<PointType>::Set_delete_criterion_param(float delete_param){
    delete_criterion_param = delete_param;
}

template <typename PointType>
void KD_TREE<PointType>::Set_balance_criterion_param(float balance_param){
    balance_criterion_param = balance_param;
}

template <typename PointType>
void KD_TREE<PointType>::set_downsample_param(float downsample_param){
    downsample_size = downsample_param;
}

template <typename PointType>
void KD_TREE<PointType>::InitializeKDTree(float delete_param, float balance_param, float box_length){
    Set_delete_criterion_param(delete_param);
    Set_balance_criterion_param(balance_param);
    set_downsample_param(box_length);
}

template <typename PointType>
void KD_TREE<PointType>::InitTreeNode(KD_TREE_NODE * root){
    root->point.x = 0.0f;
    root->point.y = 0.0f;
    root->point.z = 0.0f;       
    root->node_range_x[0] = 0.0f;
    root->node_range_x[1] = 0.0f;
    root->node_range_y[0] = 0.0f;
    root->node_range_y[1] = 0.0f;    
    root->node_range_z[0] = 0.0f;
    root->node_range_z[1] = 0.0f;     
    root->division_axis = 0;
    root->father_ptr = nullptr;
    root->left_son_ptr = nullptr;
    root->right_son_ptr = nullptr;
    root->TreeSize = 0;
    root->invalid_point_num = 0;
    root->down_del_num = 0;
    root->point_deleted = false;
    root->tree_deleted = false;
    root->need_push_down_to_left = false;
    root->need_push_down_to_right = false;
    root->point_downsample_deleted = false;
    root->working_flag = false;
    pthread_mutex_init(&(root->push_down_mutex_lock),NULL);
}   

template <typename PointType>
int KD_TREE<PointType>::size(){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr) {
            return Root_Node->TreeSize;
        } else {
            return 0;
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = Root_Node->TreeSize;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return Treesize_tmp;
        }
    }
}

template <typename PointType>
BoxPointType KD_TREE<PointType>::tree_range(){
    BoxPointType range;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr) {
            range.vertex_min[0] = Root_Node->node_range_x[0];
            range.vertex_min[1] = Root_Node->node_range_y[0];
            range.vertex_min[2] = Root_Node->node_range_z[0];
            range.vertex_max[0] = Root_Node->node_range_x[1];
            range.vertex_max[1] = Root_Node->node_range_y[1];
            range.vertex_max[2] = Root_Node->node_range_z[1];
        } else {
            memset(&range, 0, sizeof(range));
        }
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            range.vertex_min[0] = Root_Node->node_range_x[0];
            range.vertex_min[1] = Root_Node->node_range_y[0];
            range.vertex_min[2] = Root_Node->node_range_z[0];
            range.vertex_max[0] = Root_Node->node_range_x[1];
            range.vertex_max[1] = Root_Node->node_range_y[1];
            range.vertex_max[2] = Root_Node->node_range_z[1];
            pthread_mutex_unlock(&working_flag_mutex);
        } else {
            memset(&range, 0, sizeof(range));
        }
    }
    return range;
}

template <typename PointType>
int KD_TREE<PointType>::validnum(){
    int s = 0;
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        if (Root_Node != nullptr)
            return (Root_Node->TreeSize - Root_Node->invalid_point_num);
        else 
            return 0;
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            s = Root_Node->TreeSize-Root_Node->invalid_point_num;
            pthread_mutex_unlock(&working_flag_mutex);
            return s;
        } else {
            return -1;
        }
    }
}

template <typename PointType>
void KD_TREE<PointType>::root_alpha(float &alpha_bal, float &alpha_del){
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        alpha_bal = Root_Node->alpha_bal;
        alpha_del = Root_Node->alpha_del;
        return;
    } else {
        if (!pthread_mutex_trylock(&working_flag_mutex)){
            alpha_bal = Root_Node->alpha_bal;
            alpha_del = Root_Node->alpha_del;
            pthread_mutex_unlock(&working_flag_mutex);
            return;
        } else {
            alpha_bal = alpha_bal_tmp;
            alpha_del = alpha_del_tmp;      
            return;
        }
    }    
}

template <typename PointType>
void KD_TREE<PointType>::start_thread(){
    pthread_mutex_init(&termination_flag_mutex_lock, NULL);   
    pthread_mutex_init(&rebuild_ptr_mutex_lock, NULL);     
    pthread_mutex_init(&rebuild_logger_mutex_lock, NULL);
    pthread_mutex_init(&points_deleted_rebuild_mutex_lock, NULL); 
    pthread_mutex_init(&working_flag_mutex, NULL);
    pthread_mutex_init(&search_flag_mutex, NULL);
    pthread_create(&rebuild_thread, NULL, multi_thread_ptr, (void*) this);
    printf("Multi thread started \n");    
}

template <typename PointType>
void KD_TREE<PointType>::stop_thread(){
    pthread_mutex_lock(&termination_flag_mutex_lock);
    termination_flag = true;
    pthread_mutex_unlock(&termination_flag_mutex_lock);
    if (rebuild_thread) pthread_join(rebuild_thread, NULL);
    pthread_mutex_destroy(&termination_flag_mutex_lock);
    pthread_mutex_destroy(&rebuild_logger_mutex_lock);
    pthread_mutex_destroy(&rebuild_ptr_mutex_lock);
    pthread_mutex_destroy(&points_deleted_rebuild_mutex_lock);
    pthread_mutex_destroy(&working_flag_mutex);
    pthread_mutex_destroy(&search_flag_mutex);     
}

template <typename PointType>
void * KD_TREE<PointType>::multi_thread_ptr(void * arg){
    KD_TREE * handle = (KD_TREE*) arg;
    handle->multi_thread_rebuild();
    return nullptr;    
}

template <typename PointType>
/**
 * @brief 多线程重建 KD 树的函数
 * 
 * 该函数运行在一个独立的线程中，持续检查是否需要对 KD 树的某个子树进行重建。
 * 当检测到 Rebuild_Ptr 指向一个需要重建的节点时，会暂停相关操作（如搜索、删除等），
 * 然后对该子树进行序列化、重建，并将操作日志中记录的操作重新应用到新树上。
 * 最后替换旧子树为新子树，并更新相关指针和状态。
 * 
 * @note 该函数为多线程环境下的核心重建逻辑，需与搜索、插入、删除等操作互斥执行。
 * 
 * @warning 该函数依赖全局变量和互斥锁控制并发访问，不可随意更改执行顺序。
 * 
 * @return 无返回值。线程在 termination_flag 被设置为 true 后正常退出。
 */
void KD_TREE<PointType>::multi_thread_rebuild(){
    bool terminated = false;
    KD_TREE_NODE * father_ptr, ** new_node_ptr;
    
    // 检查终止标志是否被设置
    pthread_mutex_lock(&termination_flag_mutex_lock);
    terminated = termination_flag;
    pthread_mutex_unlock(&termination_flag_mutex_lock);

    while (!terminated){
        // 获取重建指针的锁，检查是否有需要重建的节点
        pthread_mutex_lock(&rebuild_ptr_mutex_lock);
        pthread_mutex_lock(&working_flag_mutex);
        if (Rebuild_Ptr != nullptr ){                    
            
            /* ==================== 遍历并复制旧子树 ==================== */
            // 检查重建日志是否为空，非空则打印错误信息（调试用途）
            if (!Rebuild_Logger.empty()){
                printf("\n\n\n\n\n\n\n\n\n\n\n ERROR!!! \n\n\n\n\n\n\n\n\n");
            }
            rebuild_flag = true;

            // 如果重建的是根节点，则保存临时参数用于后续恢复
            if (*Rebuild_Ptr == Root_Node) {
                Treesize_tmp = Root_Node->TreeSize;
                Validnum_tmp = Root_Node->TreeSize - Root_Node->invalid_point_num;
                alpha_bal_tmp = Root_Node->alpha_bal;
                alpha_del_tmp = Root_Node->alpha_del;
            }

            // 保存旧子树根节点和其父节点
            KD_TREE_NODE * old_root_node = (*Rebuild_Ptr);                            
            father_ptr = (*Rebuild_Ptr)->father_ptr;  

            // 清空重建点云存储容器
            PointVector ().swap(Rebuild_PCL_Storage);

            // 锁定搜索操作，等待所有搜索完成
            pthread_mutex_lock(&search_flag_mutex);
            while (search_mutex_counter != 0){
                pthread_mutex_unlock(&search_flag_mutex);
                usleep(1);             
                pthread_mutex_lock(&search_flag_mutex);
            }
            search_mutex_counter = -1;
            pthread_mutex_unlock(&search_flag_mutex);

            // 锁定删除点缓存，将旧子树展平为点云存储到 Rebuild_PCL_Storage 中
            pthread_mutex_lock(&points_deleted_rebuild_mutex_lock);    
            flatten(*Rebuild_Ptr, Rebuild_PCL_Storage, MULTI_THREAD_REC);
            pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);

            // 解锁搜索操作
            pthread_mutex_lock(&search_flag_mutex);
            search_mutex_counter = 0;
            pthread_mutex_unlock(&search_flag_mutex);              
            pthread_mutex_unlock(&working_flag_mutex);   

            /* ==================== 重建子树并应用操作日志 ==================== */
            Operation_Logger_Type Operation;
            KD_TREE_NODE * new_root_node = nullptr;  

            // 如果展平后的点云不为空，则开始重建
            if (int(Rebuild_PCL_Storage.size()) > 0){
                BuildTree(&new_root_node, 0, Rebuild_PCL_Storage.size()-1, Rebuild_PCL_Storage);

                // 重建完成后，将操作日志中记录的操作重新应用到新树上
                pthread_mutex_lock(&working_flag_mutex);
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                int tmp_counter = 0;
                while (!Rebuild_Logger.empty()){
                    Operation = Rebuild_Logger.front();
                    max_queue_size = max(max_queue_size, Rebuild_Logger.size());
                    Rebuild_Logger.pop();
                    pthread_mutex_unlock(&rebuild_logger_mutex_lock);                  
                    pthread_mutex_unlock(&working_flag_mutex);
                    run_operation(&new_root_node, Operation);
                    tmp_counter ++;
                    if (tmp_counter % 10 == 0) usleep(1);
                    pthread_mutex_lock(&working_flag_mutex);
                    pthread_mutex_lock(&rebuild_logger_mutex_lock);               
                }   
               pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }  

            /* ==================== 替换旧子树为新子树 ==================== */
            // 再次锁定搜索操作，等待所有搜索完成
            pthread_mutex_lock(&search_flag_mutex);
            while (search_mutex_counter != 0){
                pthread_mutex_unlock(&search_flag_mutex);
                usleep(1);             
                pthread_mutex_lock(&search_flag_mutex);
            }
            search_mutex_counter = -1;
            pthread_mutex_unlock(&search_flag_mutex);

            // 将父节点的子节点指针指向新子树
            if (father_ptr->left_son_ptr == *Rebuild_Ptr) {
                father_ptr->left_son_ptr = new_root_node;
            } else if (father_ptr->right_son_ptr == *Rebuild_Ptr){             
                father_ptr->right_son_ptr = new_root_node;
            } else {
                throw "Error: Father ptr incompatible with current node\n";
            }

            // 设置新子树的父节点指针
            if (new_root_node != nullptr) new_root_node->father_ptr = father_ptr;
            (*Rebuild_Ptr) = new_root_node;

            // 如果重建的是根节点，则更新全局根节点指针
            if (father_ptr == STATIC_ROOT_NODE) Root_Node = STATIC_ROOT_NODE->left_son_ptr;

            // 向上更新节点信息，直到根节点或遇到正在工作的节点
            KD_TREE_NODE * update_root = *Rebuild_Ptr;
            while (update_root != nullptr && update_root != Root_Node){
                update_root = update_root->father_ptr;
                if (update_root->working_flag) break;
                if (update_root == update_root->father_ptr->left_son_ptr && update_root->father_ptr->need_push_down_to_left) break;
                if (update_root == update_root->father_ptr->right_son_ptr && update_root->father_ptr->need_push_down_to_right) break;
                Update(update_root);
            }

            // 解锁搜索操作
            pthread_mutex_lock(&search_flag_mutex);
            search_mutex_counter = 0;
            pthread_mutex_unlock(&search_flag_mutex);

            // 重置重建指针和标志位
            Rebuild_Ptr = nullptr;
            pthread_mutex_unlock(&working_flag_mutex);
            rebuild_flag = false;                     

            /* ==================== 删除旧子树 ==================== */
            delete_tree_nodes(&old_root_node);
        } else {
            pthread_mutex_unlock(&working_flag_mutex);             
        }
        pthread_mutex_unlock(&rebuild_ptr_mutex_lock);         

        // 检查终止标志是否被设置
        pthread_mutex_lock(&termination_flag_mutex_lock);
        terminated = termination_flag;
        pthread_mutex_unlock(&termination_flag_mutex_lock);
        usleep(100); 
    }
    printf("Rebuild thread terminated normally\n");    
}

template <typename PointType>
void KD_TREE<PointType>::run_operation(KD_TREE_NODE ** root, Operation_Logger_Type operation){
    switch (operation.op)
    {
    case ADD_POINT:      
        Add_by_point(root, operation.point, false, (*root)->division_axis);          
        break;
    case ADD_BOX:
        Add_by_range(root, operation.boxpoint, false);
        break;
    case DELETE_POINT:
        Delete_by_point(root, operation.point, false);
        break;
    case DELETE_BOX:
        Delete_by_range(root, operation.boxpoint, false, false);
        break;
    case DOWNSAMPLE_DELETE:
        Delete_by_range(root, operation.boxpoint, false, true);
        break;
    case PUSH_DOWN:
        (*root)->tree_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->point_downsample_deleted |= operation.tree_downsample_deleted;
        (*root)->tree_deleted = operation.tree_deleted || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = (*root)->tree_deleted || (*root)->point_downsample_deleted;
        if (operation.tree_downsample_deleted) (*root)->down_del_num = (*root)->TreeSize;
        if (operation.tree_deleted) (*root)->invalid_point_num = (*root)->TreeSize;
            else (*root)->invalid_point_num = (*root)->down_del_num;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;     
        break;
    default:
        break;
    }
}

template <typename PointType>
void KD_TREE<PointType>::Build(PointVector point_cloud){
    if (Root_Node != nullptr){
        delete_tree_nodes(&Root_Node);
    }
    if (point_cloud.size() == 0) return;
    STATIC_ROOT_NODE = new KD_TREE_NODE;
    InitTreeNode(STATIC_ROOT_NODE); 
    BuildTree(&STATIC_ROOT_NODE->left_son_ptr, 0, point_cloud.size()-1, point_cloud);
    Update(STATIC_ROOT_NODE);
    STATIC_ROOT_NODE->TreeSize = 0;
    Root_Node = STATIC_ROOT_NODE->left_son_ptr;    
}

template <typename PointType>
/**
 * @brief 在KD树中搜索给定点的k个最近邻点
 * 
 * 该函数用于在KD树数据结构中查找距离给定点最近的k个点。支持设置最大搜索距离限制，
 * 并处理了多线程环境下的重建操作同步问题。
 * 
 * @param point 待搜索的查询点
 * @param k_nearest 需要查找的最近邻点的数量
 * @param Nearest_Points 输出参数，存储找到的最近邻点列表
 * @param Point_Distance 输出参数，存储对应点的距离值列表
 * @param max_dist 最大搜索距离限制，超过此距离的点将被忽略
 */
void KD_TREE<PointType>::Nearest_Search(PointType point, int k_nearest, PointVector& Nearest_Points, vector<float> & Point_Distance, double max_dist){   
    // 创建一个大小为2*k_nearest的手动堆，用于存储候选近邻点
    MANUAL_HEAP q(2*k_nearest);
    q.clear();
    // 清空距离向量，确保结果的正确性
    vector<float> ().swap(Point_Distance);
    
    // 检查是否正在进行树的重建操作
    if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
        // 无重建操作时直接进行搜索
        Search(Root_Node, k_nearest, point, q, max_dist);
    } else {
        // 存在重建操作时，需要进行线程同步
        pthread_mutex_lock(&search_flag_mutex);
        // 等待重建完成
        while (search_mutex_counter == -1)
        {
            pthread_mutex_unlock(&search_flag_mutex);
            usleep(1);
            pthread_mutex_lock(&search_flag_mutex);
        }
        // 增加搜索计数器，表示当前有搜索操作正在进行
        search_mutex_counter += 1;
        pthread_mutex_unlock(&search_flag_mutex);  
        // 执行搜索操作
        Search(Root_Node, k_nearest, point, q, max_dist);  
        // 搜索完成后减少搜索计数器
        pthread_mutex_lock(&search_flag_mutex);
        search_mutex_counter -= 1;
        pthread_mutex_unlock(&search_flag_mutex);      
    }
    
    // 确定实际找到的近邻点数量
    int k_found = min(k_nearest,int(q.size()));
    // 清空结果向量并重新分配空间
    PointVector ().swap(Nearest_Points);
    vector<float> ().swap(Point_Distance);
    
    // 从堆中提取k个最近的点和对应距离
    for (int i=0;i < k_found;i++){
        Nearest_Points.insert(Nearest_Points.begin(), q.top().point);
        Point_Distance.insert(Point_Distance.begin(), q.top().dist);
        q.pop();
    }
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Box_Search(const BoxPointType &Box_of_Point, PointVector &Storage)
{
    Storage.clear();
    Search_by_range(Root_Node, Box_of_Point, Storage);
}

template <typename PointType>
void KD_TREE<PointType>::Radius_Search(PointType point, const float radius, PointVector &Storage)
{
    Storage.clear();
    Search_by_radius(Root_Node, point, radius, Storage);
}


/**
 * @brief 向KD树中添加一组点，并根据下采样开关决定是否进行下采样处理。
 * 
 * 该函数遍历输入的点集，若启用下采样，则对每个点所在的体素范围进行搜索，
 * 找到距离体素中心最近的点作为代表点插入树中；否则直接将点插入树中。
 * 插入过程中会判断是否正在进行重建操作，以决定是否需要记录操作日志。
 *
 * @param PointToAdd 待添加的点集合（PointVector类型）
 * @param downsample_on 是否启用下采样的标志位
 * @return 返回实际被插入或更新的点的数量
 */
template <typename PointType>
int KD_TREE<PointType>::Add_Points(PointVector & PointToAdd, bool downsample_on){
    int NewPointSize = PointToAdd.size();
    int tree_size = size();
    BoxPointType Box_of_Point;
    PointType downsample_result, mid_point;
    bool downsample_switch = downsample_on && DOWNSAMPLE_SWITCH;
    float min_dist, tmp_dist;
    int tmp_counter = 0;

    // 遍历所有待添加的点
    for (int i=0; i<PointToAdd.size();i++){
        if (downsample_switch){
            // 计算当前点所在体素的边界框
            Box_of_Point.vertex_min[0] = floor(PointToAdd[i].x/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[0] = Box_of_Point.vertex_min[0]+downsample_size;
            Box_of_Point.vertex_min[1] = floor(PointToAdd[i].y/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[1] = Box_of_Point.vertex_min[1]+downsample_size; 
            Box_of_Point.vertex_min[2] = floor(PointToAdd[i].z/downsample_size)*downsample_size;
            Box_of_Point.vertex_max[2] = Box_of_Point.vertex_min[2]+downsample_size;   

            // 计算体素中心点坐标
            mid_point.x = Box_of_Point.vertex_min[0] + (Box_of_Point.vertex_max[0]-Box_of_Point.vertex_min[0])/2.0;
            mid_point.y = Box_of_Point.vertex_min[1] + (Box_of_Point.vertex_max[1]-Box_of_Point.vertex_min[1])/2.0;
            mid_point.z = Box_of_Point.vertex_min[2] + (Box_of_Point.vertex_max[2]-Box_of_Point.vertex_min[2])/2.0;

            // 清空临时存储并查找当前体素内的所有点
            PointVector ().swap(Downsample_Storage);
            Search_by_range(Root_Node, Box_of_Point, Downsample_Storage);

            // 在体素内寻找距离中心最近的点
            min_dist = calc_dist(PointToAdd[i],mid_point);
            downsample_result = PointToAdd[i];                
            for (int index = 0; index < Downsample_Storage.size(); index++){
                tmp_dist = calc_dist(Downsample_Storage[index], mid_point);
                if (tmp_dist < min_dist){
                    min_dist = tmp_dist;
                    downsample_result = Downsample_Storage[index];
                }
            }

            // 判断是否处于重建状态，选择不同的插入方式
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){  
                if (Downsample_Storage.size() > 1 || same_point(PointToAdd[i], downsample_result)){
                    if (Downsample_Storage.size() > 0) Delete_by_range(&Root_Node, Box_of_Point, true, true);
                    Add_by_point(&Root_Node, downsample_result, true, Root_Node->division_axis);
                    tmp_counter ++;                      
                }
            } else {
                if (Downsample_Storage.size() > 1 || same_point(PointToAdd[i], downsample_result)){
                    Operation_Logger_Type  operation_delete, operation;
                    operation_delete.boxpoint = Box_of_Point;
                    operation_delete.op = DOWNSAMPLE_DELETE;
                    operation.point = downsample_result;
                    operation.op = ADD_POINT;

                    pthread_mutex_lock(&working_flag_mutex);
                    if (Downsample_Storage.size() > 0) Delete_by_range(&Root_Node, Box_of_Point, false , true);                                      
                    Add_by_point(&Root_Node, downsample_result, false, Root_Node->division_axis);
                    tmp_counter ++;

                    // 如果正在重建，则记录操作日志
                    if (rebuild_flag){
                        pthread_mutex_lock(&rebuild_logger_mutex_lock);
                        if (Downsample_Storage.size() > 0) Rebuild_Logger.push(operation_delete);
                        Rebuild_Logger.push(operation);
                        pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                    }
                    pthread_mutex_unlock(&working_flag_mutex);
                };
            }
        } else {
            // 不启用下采样时直接插入点
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
                Add_by_point(&Root_Node, PointToAdd[i], true, Root_Node->division_axis);     
            } else {
                Operation_Logger_Type operation;
                operation.point = PointToAdd[i];
                operation.op = ADD_POINT;                

                pthread_mutex_lock(&working_flag_mutex);
                Add_by_point(&Root_Node, PointToAdd[i], false, Root_Node->division_axis);
                if (rebuild_flag){
                    pthread_mutex_lock(&rebuild_logger_mutex_lock);
                    Rebuild_Logger.push(operation);
                    pthread_mutex_unlock(&rebuild_logger_mutex_lock);
                }
                pthread_mutex_unlock(&working_flag_mutex);       
            }
        }
    }
    return tmp_counter;
}

template <typename PointType>
void KD_TREE<PointType>::Add_Point_Boxes(vector<BoxPointType> & BoxPoints){     
    for (int i=0;i < BoxPoints.size();i++){
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){
            Add_by_range(&Root_Node ,BoxPoints[i], true);
        } else {
            Operation_Logger_Type operation;
            operation.boxpoint = BoxPoints[i];
            operation.op = ADD_BOX;
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_range(&Root_Node ,BoxPoints[i], false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }               
            pthread_mutex_unlock(&working_flag_mutex);
        }    
    } 
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Delete_Points(PointVector & PointToDel){        
    for (int i=0;i<PointToDel.size();i++){
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){               
            Delete_by_point(&Root_Node, PointToDel[i], true);
        } else {
            Operation_Logger_Type operation;
            operation.point = PointToDel[i];
            operation.op = DELETE_POINT;
            pthread_mutex_lock(&working_flag_mutex);        
            Delete_by_point(&Root_Node, PointToDel[i], false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }      
    }      
    return;
}

/**
 * @brief 从KD树中删除指定包围盒范围内的点
 * 
 * 该函数遍历输入的包围盒列表，删除每个包围盒范围内的所有点。
 * 根据重建指针的状态，选择直接删除或在重建过程中记录删除操作。
 * 
 * @param BoxPoints 包含多个包围盒的向量，每个包围盒定义了要删除点的空间范围
 * @return 返回实际删除的点的数量
 */
template <typename PointType>
int KD_TREE<PointType>::Delete_Point_Boxes(vector<BoxPointType> & BoxPoints){
    int tmp_counter = 0;
    
    // 遍历所有包围盒，逐个执行删除操作
    for (int i=0;i < BoxPoints.size();i++){ 
        // 检查是否处于重建状态
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node){       
            // 非重建状态：直接执行删除操作
            tmp_counter += Delete_by_range(&Root_Node ,BoxPoints[i], true, false);
        } else {
            cout << "Delete_Point_Boxes  Rebuild_Ptr == nullptr || *Rebuild_Ptr != Root_Node)" << endl;
            // 重建状态：需要记录操作日志
            Operation_Logger_Type operation;
            operation.boxpoint = BoxPoints[i];
            operation.op = DELETE_BOX;     
            
            pthread_mutex_lock(&working_flag_mutex); 
            // 执行删除但不立即更新树结构
            tmp_counter += Delete_by_range(&Root_Node ,BoxPoints[i], false, false);
            
            // 如果正在重建，则将删除操作记录到重建日志中
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }                
            pthread_mutex_unlock(&working_flag_mutex);
        }
    } 
    
    return tmp_counter;
}

template <typename PointType>
void KD_TREE<PointType>::acquire_removed_points(PointVector & removed_points){
    pthread_mutex_lock(&points_deleted_rebuild_mutex_lock); 
    for (int i = 0; i < Points_deleted.size();i++){
        removed_points.push_back(Points_deleted[i]);
    }
    for (int i = 0; i < Multithread_Points_deleted.size();i++){
        removed_points.push_back(Multithread_Points_deleted[i]);
    }
    Points_deleted.clear();
    Multithread_Points_deleted.clear();
    pthread_mutex_unlock(&points_deleted_rebuild_mutex_lock);   
    return;
}

template <typename PointType>
/**
 * @brief 递归构建KD树
 * 
 * 该函数通过递归方式构建一个平衡的KD树。它会选择最佳分割轴（最长维度），并使用nth_element进行
 * 中位数划分，然后分别递归构建左右子树。
 * 
 * @param root 指向当前节点指针的指针，用于构建新的树节点
 * @param l 当前处理的数据范围的左边界（包含）
 * @param r 当前处理的数据范围的右边界（包含）
 * @param Storage 存储所有点的向量，构建过程中会对其中的元素进行重排
 */
void KD_TREE<PointType>::BuildTree(KD_TREE_NODE ** root, int l, int r, PointVector & Storage){
    // 如果左边界大于右边界，则没有数据需要处理，直接返回
    if (l>r) return;
    
    // 创建新节点并初始化
    *root = new KD_TREE_NODE;
    InitTreeNode(*root);
    
    // 计算中位数索引
    int mid = (l+r)>>1;
    int div_axis = 0;
    int i;
    
    // 遍历当前范围内的所有点，计算每个维度的最小值和最大值
    float min_value[3] = {INFINITY, INFINITY, INFINITY};
    float max_value[3] = {-INFINITY, -INFINITY, -INFINITY};
    float dim_range[3] = {0,0,0};
    for (i=l;i<=r;i++){
        min_value[0] = min(min_value[0], Storage[i].x);
        min_value[1] = min(min_value[1], Storage[i].y);
        min_value[2] = min(min_value[2], Storage[i].z);
        max_value[0] = max(max_value[0], Storage[i].x);
        max_value[1] = max(max_value[1], Storage[i].y);
        max_value[2] = max(max_value[2], Storage[i].z);
    }
    
    // 计算每个维度的范围，并选择范围最大的维度作为分割轴
    for (i=0;i<3;i++) dim_range[i] = max_value[i] - min_value[i];
    for (i=1;i<3;i++) if (dim_range[i] > dim_range[div_axis]) div_axis = i;
    
    // 根据选定的分割轴对数据进行排序，将中位数放在mid位置
    (*root)->division_axis = div_axis;
    switch (div_axis)
    {
    case 0:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    case 1:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_y);
        break;
    case 2:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_z);
        break;
    default:
        nth_element(begin(Storage)+l, begin(Storage)+mid, begin(Storage)+r+1, point_cmp_x);
        break;
    }  
    
    // 将中位数点设为当前节点的值
    (*root)->point = Storage[mid]; 
    
    // 递归构建左右子树
    KD_TREE_NODE * left_son = nullptr, * right_son = nullptr;
    BuildTree(&left_son, l, mid-1, Storage);
    BuildTree(&right_son, mid+1, r, Storage);  
    
    // 设置左右子节点指针
    (*root)->left_son_ptr = left_son;
    (*root)->right_son_ptr = right_son;
    
    // 更新当前节点的信息（如包围盒等）
    Update((*root));  
    
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Rebuild(KD_TREE_NODE ** root){    
    KD_TREE_NODE * father_ptr;
    if ((*root)->TreeSize >= Multi_Thread_Rebuild_Point_Num) { 
        if (!pthread_mutex_trylock(&rebuild_ptr_mutex_lock)){     
            if (Rebuild_Ptr == nullptr || ((*root)->TreeSize > (*Rebuild_Ptr)->TreeSize)) {
                Rebuild_Ptr = root;          
            }
            pthread_mutex_unlock(&rebuild_ptr_mutex_lock);
        }
    } else {
        father_ptr = (*root)->father_ptr;
        int size_rec = (*root)->TreeSize;
        PCL_Storage.clear();
        flatten(*root, PCL_Storage, DELETE_POINTS_REC);
        delete_tree_nodes(root);
        BuildTree(root, 0, PCL_Storage.size()-1, PCL_Storage);
        if (*root != nullptr) (*root)->father_ptr = father_ptr;
        if (*root == Root_Node) STATIC_ROOT_NODE->left_son_ptr = *root;
    } 
    return;
}

template <typename PointType>
/*
 * 函数名: Delete_by_range
 * 功能: 在KD树中删除指定范围内的点。支持普通删除和下采样删除两种模式。
 * 参数:
 *   root              - 指向当前节点指针的指针，用于递归操作树结构
 *   boxpoint          - 删除范围的边界框（BoxPointType类型）
 *   allow_rebuild     - 是否允许在删除后进行树的重建优化
 *   is_downsample     - 是否为下采样模式下的删除操作
 * 返回值: 被删除的点的数量
 *
 * 说明:
 *   该函数通过递归遍历KD树，判断节点是否落在给定的删除范围内，并标记删除。
 *   如果整个子树都落在删除范围内，则直接标记整棵子树为删除状态；
 *   否则继续递归处理左右子树，并在回溯时更新节点信息。
 *   支持多线程环境下的安全操作与重建日志记录。
 */
int KD_TREE<PointType>::Delete_by_range(KD_TREE_NODE ** root,  BoxPointType boxpoint, bool allow_rebuild, bool is_downsample){   
    // 如果当前节点为空或已被删除，直接返回
    if ((*root) == nullptr || (*root)->tree_deleted) return 0;

    // 标记当前节点正在被操作
    (*root)->working_flag = true;
    Push_Down(*root);  // 将父节点的删除状态向下传递

    int tmp_counter = 0;

    // 判断当前节点的x、y、z范围是否与删除范围有交集，无交集则返回
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1]) return 0;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1]) return 0;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1]) return 0;

    // 如果当前节点的整个范围完全包含在删除范围内
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] &&
        boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1] > (*root)->node_range_y[1] &&
        boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1]) {

        // 标记整棵树为删除状态
        (*root)->tree_deleted = true;
        (*root)->point_deleted = true;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;

        // 计算有效点数量并更新无效点计数
        tmp_counter = (*root)->TreeSize - (*root)->invalid_point_num;
        (*root)->invalid_point_num = (*root)->TreeSize;

        // 如果是下采样删除，也标记下采样删除状态
        if (is_downsample){
            (*root)->tree_downsample_deleted = true;
            (*root)->point_downsample_deleted = true;
            (*root)->down_del_num = (*root)->TreeSize;
        }

        return tmp_counter;
    }

    // 如果当前节点的点在删除范围内且未被删除，则标记该点为删除
    if (!(*root)->point_deleted &&
        boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x &&
        boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y &&
        boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z) {

        (*root)->point_deleted = true;
        tmp_counter += 1;

        if (is_downsample) (*root)->point_downsample_deleted = true;
    }

    // 构造操作日志对象
    Operation_Logger_Type delete_box_log;
    struct timespec Timeout;    

    // 根据删除类型设置操作类型
    if (is_downsample)
        delete_box_log.op = DOWNSAMPLE_DELETE;
    else
        delete_box_log.op = DELETE_BOX;

    delete_box_log.boxpoint = boxpoint;

    // 处理左子树
    if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){
        tmp_counter += Delete_by_range(&((*root)->left_son_ptr), boxpoint, allow_rebuild, is_downsample);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        tmp_counter += Delete_by_range(&((*root)->left_son_ptr), boxpoint, false, is_downsample);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(delete_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }

    // 处理右子树
    if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){
        tmp_counter += Delete_by_range(&((*root)->right_son_ptr), boxpoint, allow_rebuild, is_downsample);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        tmp_counter += Delete_by_range(&((*root)->right_son_ptr), boxpoint, false, is_downsample);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(delete_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }    

    // 更新当前节点的状态信息
    Update(*root);     

    // 如果当前节点是重建指针指向的节点，并且节点规模小于阈值，则清除重建指针
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) 
    {
        Rebuild_Ptr = nullptr; 
        cout << "Clear Rebuild_Ptr" << endl;
    }
        

    // 检查是否需要重建树结构
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);

    // 清除当前节点的工作标志
    if ((*root) != nullptr) (*root)->working_flag = false;

    return tmp_counter;
}


template <typename PointType>
void KD_TREE<PointType>::Delete_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild){   
    if ((*root) == nullptr || (*root)->tree_deleted) return;
    (*root)->working_flag = true;
    Push_Down(*root);
    if (same_point((*root)->point, point) && !(*root)->point_deleted) {          
        (*root)->point_deleted = true;
        (*root)->invalid_point_num += 1;
        if ((*root)->invalid_point_num == (*root)->TreeSize) (*root)->tree_deleted = true;    
        return;
    }
    Operation_Logger_Type delete_log;
    struct timespec Timeout;    
    delete_log.op = DELETE_POINT;
    delete_log.point = point;     
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z)){           
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            Delete_by_point(&(*root)->left_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Delete_by_point(&(*root)->left_son_ptr, point,false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }
    } else {       
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Delete_by_point(&(*root)->right_son_ptr, point, allow_rebuild);         
        } else {
            pthread_mutex_lock(&working_flag_mutex); 
            Delete_by_point(&(*root)->right_son_ptr, point, false);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(delete_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);
        }        
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Add_by_range(KD_TREE_NODE ** root, BoxPointType boxpoint, bool allow_rebuild){
    if ((*root) == nullptr) return;
    (*root)->working_flag = true;
    Push_Down(*root);       
    if (boxpoint.vertex_max[0] <= (*root)->node_range_x[0] || boxpoint.vertex_min[0] > (*root)->node_range_x[1]) return;
    if (boxpoint.vertex_max[1] <= (*root)->node_range_y[0] || boxpoint.vertex_min[1] > (*root)->node_range_y[1]) return;
    if (boxpoint.vertex_max[2] <= (*root)->node_range_z[0] || boxpoint.vertex_min[2] > (*root)->node_range_z[1]) return;
    if (boxpoint.vertex_min[0] <= (*root)->node_range_x[0] && boxpoint.vertex_max[0] > (*root)->node_range_x[1] && boxpoint.vertex_min[1] <= (*root)->node_range_y[0] && boxpoint.vertex_max[1]> (*root)->node_range_y[1] && boxpoint.vertex_min[2] <= (*root)->node_range_z[0] && boxpoint.vertex_max[2] > (*root)->node_range_z[1]){
        (*root)->tree_deleted = false || (*root)->tree_downsample_deleted;
        (*root)->point_deleted = false || (*root)->point_downsample_deleted;
        (*root)->need_push_down_to_left = true;
        (*root)->need_push_down_to_right = true;
        (*root)->invalid_point_num = (*root)->down_del_num; 
        return;
    }
    if (boxpoint.vertex_min[0] <= (*root)->point.x && boxpoint.vertex_max[0] > (*root)->point.x && boxpoint.vertex_min[1] <= (*root)->point.y && boxpoint.vertex_max[1] > (*root)->point.y && boxpoint.vertex_min[2] <= (*root)->point.z && boxpoint.vertex_max[2] > (*root)->point.z){
        (*root)->point_deleted = (*root)->point_downsample_deleted;
    }
    Operation_Logger_Type add_box_log;
    struct timespec Timeout;    
    add_box_log.op = ADD_BOX;
    add_box_log.boxpoint = boxpoint;
    if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){
        Add_by_range(&((*root)->left_son_ptr), boxpoint, allow_rebuild);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        Add_by_range(&((*root)->left_son_ptr), boxpoint, false);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(add_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }        
        pthread_mutex_unlock(&working_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){
        Add_by_range(&((*root)->right_son_ptr), boxpoint, allow_rebuild);
    } else {
        pthread_mutex_lock(&working_flag_mutex);
        Add_by_range(&((*root)->right_son_ptr), boxpoint, false);
        if (rebuild_flag){
            pthread_mutex_lock(&rebuild_logger_mutex_lock);
            Rebuild_Logger.push(add_box_log);
            pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
        }
        pthread_mutex_unlock(&working_flag_mutex);
    }
    Update(*root);
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root);
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Add_by_point(KD_TREE_NODE ** root, PointType point, bool allow_rebuild, int father_axis){     
    if (*root == nullptr){
        *root = new KD_TREE_NODE;
        InitTreeNode(*root);
        (*root)->point = point;
        (*root)->division_axis = (father_axis + 1) % 3;
        Update(*root);
        return;
    }
    (*root)->working_flag = true;
    Operation_Logger_Type add_log;
    struct timespec Timeout;    
    add_log.op = ADD_POINT;
    add_log.point = point;
    Push_Down(*root);
    if (((*root)->division_axis == 0 && point.x < (*root)->point.x) || ((*root)->division_axis == 1 && point.y < (*root)->point.y) || ((*root)->division_axis == 2 && point.z < (*root)->point.z)){
        if ((Rebuild_Ptr == nullptr) || (*root)->left_son_ptr != *Rebuild_Ptr){          
            Add_by_point(&(*root)->left_son_ptr, point, allow_rebuild, (*root)->division_axis);
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->left_son_ptr, point, false,(*root)->division_axis);
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex);            
        }
    } else {  
        if ((Rebuild_Ptr == nullptr) || (*root)->right_son_ptr != *Rebuild_Ptr){         
            Add_by_point(&(*root)->right_son_ptr, point, allow_rebuild,(*root)->division_axis);
        } else {
            pthread_mutex_lock(&working_flag_mutex);
            Add_by_point(&(*root)->right_son_ptr, point, false,(*root)->division_axis);       
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(add_log);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);                 
            }
            pthread_mutex_unlock(&working_flag_mutex); 
        }
    }
    Update(*root);   
    if (Rebuild_Ptr != nullptr && *Rebuild_Ptr == *root && (*root)->TreeSize < Multi_Thread_Rebuild_Point_Num) Rebuild_Ptr = nullptr; 
    bool need_rebuild = allow_rebuild & Criterion_Check((*root));
    if (need_rebuild) Rebuild(root); 
    if ((*root) != nullptr) (*root)->working_flag = false;   
    return;
}

template <typename PointType>
void KD_TREE<PointType>::Search(KD_TREE_NODE * root, int k_nearest, PointType point, MANUAL_HEAP &q, double max_dist){
    if (root == nullptr || root->tree_deleted) return;   
    double cur_dist = calc_box_dist(root, point);
    double max_dist_sqr = max_dist * max_dist;
    if (cur_dist > max_dist_sqr) return;    
    int retval; 
    if (root->need_push_down_to_left || root->need_push_down_to_right) {
        retval = pthread_mutex_trylock(&(root->push_down_mutex_lock));
        if (retval == 0){
            Push_Down(root);
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        } else {
            pthread_mutex_lock(&(root->push_down_mutex_lock));
            pthread_mutex_unlock(&(root->push_down_mutex_lock));
        }
    }
    if (!root->point_deleted){
        float dist = calc_dist(point, root->point);
        if (dist <= max_dist_sqr && (q.size() < k_nearest || dist < q.top().dist)){
            if (q.size() >= k_nearest) q.pop();
            PointType_CMP current_point{root->point, dist};                    
            q.push(current_point);            
        }
    }  
    int cur_search_counter;
    float dist_left_node = calc_box_dist(root->left_son_ptr, point);
    float dist_right_node = calc_box_dist(root->right_son_ptr, point);
    if (q.size()< k_nearest || dist_left_node < q.top().dist && dist_right_node < q.top().dist){
        if (dist_left_node <= dist_right_node) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_right_node < q.top().dist) {
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                    Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);                    
                    Search(root->right_son_ptr, k_nearest, point, q, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }                
            }
        } else {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);                   
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
            if (q.size() < k_nearest || dist_left_node < q.top().dist) {            
                if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                    Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
                } else {
                    pthread_mutex_lock(&search_flag_mutex);
                    while (search_mutex_counter == -1)
                    {
                        pthread_mutex_unlock(&search_flag_mutex);
                        usleep(1);
                        pthread_mutex_lock(&search_flag_mutex);
                    }
                    search_mutex_counter += 1;
                    pthread_mutex_unlock(&search_flag_mutex);  
                    Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                    pthread_mutex_lock(&search_flag_mutex);
                    search_mutex_counter -= 1;
                    pthread_mutex_unlock(&search_flag_mutex);
                }
            }
        }
    } else {
        if (dist_left_node < q.top().dist) {        
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->left_son_ptr, k_nearest, point, q, max_dist);  
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
        if (dist_right_node < q.top().dist) {
            if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);                       
            } else {
                pthread_mutex_lock(&search_flag_mutex);
                while (search_mutex_counter == -1)
                {
                    pthread_mutex_unlock(&search_flag_mutex);
                    usleep(1);
                    pthread_mutex_lock(&search_flag_mutex);
                }
                search_mutex_counter += 1;
                pthread_mutex_unlock(&search_flag_mutex);  
                Search(root->right_son_ptr, k_nearest, point, q, max_dist);
                pthread_mutex_lock(&search_flag_mutex);
                search_mutex_counter -= 1;
                pthread_mutex_unlock(&search_flag_mutex);
            }
        }
    }
    return;
}

/**
 * @brief 在KD树中搜索指定范围内的所有点
 * 
 * 该函数通过递归遍历KD树，查找所有位于给定包围盒(boxpoint)内的点，并将这些点存储到Storage中。
 * 搜索过程中会跳过被删除的节点，并处理重建过程中的线程安全问题。
 * 
 * @tparam PointType 点的类型，需包含x, y, z坐标字段
 * @param root KD树的根节点指针
 * @param boxpoint 搜索范围的包围盒，包含vertex_min和vertex_max两个顶点
 * @param Storage 用于存储搜索结果的点向量引用
 */
template <typename PointType>
void KD_TREE<PointType>::Search_by_range(KD_TREE_NODE *root, BoxPointType boxpoint, PointVector & Storage){
    // 空节点直接返回
    if (root == nullptr) return;
    
    // 下推操作，确保节点状态正确
    Push_Down(root);       
    
    // 检查包围盒在X轴上是否与节点范围有交集，无交集则返回
    if (boxpoint.vertex_max[0] <= root->node_range_x[0] || boxpoint.vertex_min[0] > root->node_range_x[1]) return;
    // 检查包围盒在Y轴上是否与节点范围有交集，无交集则返回
    if (boxpoint.vertex_max[1] <= root->node_range_y[0] || boxpoint.vertex_min[1] > root->node_range_y[1]) return;
    // 检查包围盒在Z轴上是否与节点范围有交集，无交集则返回
    if (boxpoint.vertex_max[2] <= root->node_range_z[0] || boxpoint.vertex_min[2] > root->node_range_z[1]) return;
    
    // 如果包围盒完全包含当前节点的范围，则将整个子树展平添加到结果中
    if (boxpoint.vertex_min[0] <= root->node_range_x[0] && boxpoint.vertex_max[0] > root->node_range_x[1] && boxpoint.vertex_min[1] <= root->node_range_y[0] && boxpoint.vertex_max[1] > root->node_range_y[1] && boxpoint.vertex_min[2] <= root->node_range_z[0] && boxpoint.vertex_max[2] > root->node_range_z[1]){
        flatten(root, Storage, NOT_RECORD);
        return;
    }
    
    // 检查当前节点的点是否在搜索范围内，如果在且未被删除则加入结果
    if (boxpoint.vertex_min[0] <= root->point.x && boxpoint.vertex_max[0] > root->point.x && boxpoint.vertex_min[1] <= root->point.y && boxpoint.vertex_max[1] > root->point.y && boxpoint.vertex_min[2] <= root->point.z && boxpoint.vertex_max[2] > root->point.z){
        if (!root->point_deleted) Storage.push_back(root->point);
    }
    
    // 递归搜索左子树，处理重建时的线程安全问题
    if ((Rebuild_Ptr == nullptr) || root->left_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->left_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    
    // 递归搜索右子树，处理重建时的线程安全问题
    if ((Rebuild_Ptr == nullptr) || root->right_son_ptr != *Rebuild_Ptr){
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
    } else {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_range(root->right_son_ptr, boxpoint, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    return;    
}

template <typename PointType>
void KD_TREE<PointType>::Search_by_radius(KD_TREE_NODE *root, PointType point, float radius, PointVector &Storage)
{
    if (root == nullptr)
        return;
    Push_Down(root);
    PointType range_center;
    range_center.x = (root->node_range_x[0] + root->node_range_x[1]) * 0.5;
    range_center.y = (root->node_range_y[0] + root->node_range_y[1]) * 0.5;
    range_center.z = (root->node_range_z[0] + root->node_range_z[1]) * 0.5;
    float dist = sqrt(calc_dist(range_center, point));
    if (dist > radius + sqrt(root->radius_sq)) return;
    if (dist <= radius - sqrt(root->radius_sq)) 
    {
        flatten(root, Storage, NOT_RECORD);
        return;
    }
    if (!root->point_deleted && calc_dist(root->point, point) <= radius * radius){
        Storage.push_back(root->point);
    }
    if ((Rebuild_Ptr == nullptr) || root->left_son_ptr != *Rebuild_Ptr)
    {
        Search_by_radius(root->left_son_ptr, point, radius, Storage);
    }
    else
    {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_radius(root->left_son_ptr, point, radius, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }
    if ((Rebuild_Ptr == nullptr) || root->right_son_ptr != *Rebuild_Ptr)
    {
        Search_by_radius(root->right_son_ptr, point, radius, Storage);
    }
    else
    {
        pthread_mutex_lock(&search_flag_mutex);
        Search_by_radius(root->right_son_ptr, point, radius, Storage);
        pthread_mutex_unlock(&search_flag_mutex);
    }    
    return;
}

/**
 * @brief 检查KD树节点是否需要重新平衡的判据函数
 * 
 * 该函数通过评估节点的删除率和平衡因子来判断当前节点是否需要进行重新平衡操作。
 * 主要评估两个指标：1）无效点的比例 2）左右子树的平衡度
 * 
 * @param root 指向需要检查的KD树节点的指针
 * @return bool 返回true表示需要重新平衡，返回false表示不需要重新平衡
 */
template <typename PointType>
bool KD_TREE<PointType>::Criterion_Check(KD_TREE_NODE * root){
    // 如果树的大小小于最小不平衡树大小阈值，则不需要重新平衡
    if (root->TreeSize <= Minimal_Unbalanced_Tree_Size){
        return false;
    }
    float balance_evaluation = 0.0f;
    float delete_evaluation = 0.0f;
    KD_TREE_NODE * son_ptr = root->left_son_ptr;
    // 获取存在的子节点指针，优先选择左子节点，如果为空则选择右子节点
    if (son_ptr == nullptr) son_ptr = root->right_son_ptr;
    // 计算删除评估值：无效点数量占总节点数的比例
    delete_evaluation = float(root->invalid_point_num)/ root->TreeSize;
    // 计算平衡评估值：子树大小占父节点有效节点数的比例
    balance_evaluation = float(son_ptr->TreeSize) / (root->TreeSize-1);  
    // 如果删除率超过删除判据参数，则需要重新平衡
    if (delete_evaluation > delete_criterion_param){
        return true;
    }
    // 如果平衡因子超过平衡判据参数的范围，则需要重新平衡
    if (balance_evaluation > balance_criterion_param || balance_evaluation < 1-balance_criterion_param){
        return true;
    } 
    return false;
}

/**
 * @brief 将当前节点的删除标记向下传递给其左右子节点（Push Down 操作）
 * 
 * 该函数用于 KD 树中，将某个节点上的删除状态（包括普通删除和下采样删除）传递到其子节点。
 * 如果子节点正在重建，则需要加锁处理，并可能记录操作日志。
 * 
 * @tparam PointType 点类型模板参数
 * @param root 当前处理的树节点指针
 * 
 * @note 此函数不返回任何值
 */
template <typename PointType>
void KD_TREE<PointType>::Push_Down(KD_TREE_NODE *root){
    if (root == nullptr) return;

    // 构造一个操作日志对象，记录本次 push down 操作
    Operation_Logger_Type operation;
    operation.op = PUSH_DOWN;
    operation.tree_deleted = root->tree_deleted;
    operation.tree_downsample_deleted = root->tree_downsample_deleted;

    // 处理左子节点的 push down 逻辑
    if (root->need_push_down_to_left && root->left_son_ptr != nullptr){
        // 判断左子节点是否正在重建
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->left_son_ptr){
            // 不在重建时直接更新左子节点的状态
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;

            // 更新无效点数量和删除计数
            if (root->tree_downsample_deleted)
                root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
            else
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;

            // 标记左右子节点也需要 push down
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;
            root->need_push_down_to_left = false;
        } else {
            // 左子节点正在重建，需要加锁处理
            pthread_mutex_lock(&working_flag_mutex);

            // 更新左子节点状态
            root->left_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->left_son_ptr->tree_deleted = root->tree_deleted || root->left_son_ptr->tree_downsample_deleted;
            root->left_son_ptr->point_deleted = root->left_son_ptr->tree_deleted || root->left_son_ptr->point_downsample_deleted;

            // 更新无效点数量和删除计数
            if (root->tree_downsample_deleted)
                root->left_son_ptr->down_del_num = root->left_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->TreeSize;
            else
                root->left_son_ptr->invalid_point_num = root->left_son_ptr->down_del_num;

            // 标记左右子节点也需要 push down
            root->left_son_ptr->need_push_down_to_left = true;
            root->left_son_ptr->need_push_down_to_right = true;

            // 如果开启了 rebuild 日志记录，则记录当前操作
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }

            root->need_push_down_to_left = false;
            pthread_mutex_unlock(&working_flag_mutex);
        }
    }

    // 处理右子节点的 push down 逻辑
    if (root->need_push_down_to_right && root->right_son_ptr != nullptr){
        // 判断右子节点是否正在重建
        if (Rebuild_Ptr == nullptr || *Rebuild_Ptr != root->right_son_ptr){
            // 不在重建时直接更新右子节点的状态
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;

            // 更新无效点数量和删除计数
            if (root->tree_downsample_deleted)
                root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
            else
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;

            // 标记左右子节点也需要 push down
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;
            root->need_push_down_to_right = false;
        } else {
            // 右子节点正在重建，需要加锁处理
            pthread_mutex_lock(&working_flag_mutex);

            // 更新右子节点状态
            root->right_son_ptr->tree_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->point_downsample_deleted |= root->tree_downsample_deleted;
            root->right_son_ptr->tree_deleted = root->tree_deleted || root->right_son_ptr->tree_downsample_deleted;
            root->right_son_ptr->point_deleted = root->right_son_ptr->tree_deleted || root->right_son_ptr->point_downsample_deleted;

            // 更新无效点数量和删除计数
            if (root->tree_downsample_deleted)
                root->right_son_ptr->down_del_num = root->right_son_ptr->TreeSize;
            if (root->tree_deleted)
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->TreeSize;
            else
                root->right_son_ptr->invalid_point_num = root->right_son_ptr->down_del_num;

            // 标记左右子节点也需要 push down
            root->right_son_ptr->need_push_down_to_left = true;
            root->right_son_ptr->need_push_down_to_right = true;

            // 如果开启了 rebuild 日志记录，则记录当前操作
            if (rebuild_flag){
                pthread_mutex_lock(&rebuild_logger_mutex_lock);
                Rebuild_Logger.push(operation);
                pthread_mutex_unlock(&rebuild_logger_mutex_lock);
            }

            root->need_push_down_to_right = false;
            pthread_mutex_unlock(&working_flag_mutex);
        }
    }

    return;
}

template <typename PointType>
void KD_TREE<PointType>::Update(KD_TREE_NODE * root){
    KD_TREE_NODE * left_son_ptr = root->left_son_ptr;
    KD_TREE_NODE * right_son_ptr = root->right_son_ptr;
    float tmp_range_x[2] = {INFINITY, -INFINITY};
    float tmp_range_y[2] = {INFINITY, -INFINITY};
    float tmp_range_z[2] = {INFINITY, -INFINITY};
    // Update Tree Size   
    if (left_son_ptr != nullptr && right_son_ptr != nullptr){
        root->TreeSize = left_son_ptr->TreeSize + right_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);
        root->down_del_num = left_son_ptr->down_del_num + right_son_ptr->down_del_num + (root->point_downsample_deleted? 1:0);
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = left_son_ptr->tree_deleted && right_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !right_son_ptr->tree_deleted && !root->point_deleted)){
            tmp_range_x[0] = min(min(left_son_ptr->node_range_x[0],right_son_ptr->node_range_x[0]),root->point.x);
            tmp_range_x[1] = max(max(left_son_ptr->node_range_x[1],right_son_ptr->node_range_x[1]),root->point.x);
            tmp_range_y[0] = min(min(left_son_ptr->node_range_y[0],right_son_ptr->node_range_y[0]),root->point.y);
            tmp_range_y[1] = max(max(left_son_ptr->node_range_y[1],right_son_ptr->node_range_y[1]),root->point.y);
            tmp_range_z[0] = min(min(left_son_ptr->node_range_z[0],right_son_ptr->node_range_z[0]),root->point.z);
            tmp_range_z[1] = max(max(left_son_ptr->node_range_z[1],right_son_ptr->node_range_z[1]),root->point.z);
        } else {
            if (!left_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], left_son_ptr->node_range_z[1]);
            }
            if (!right_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], right_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }
        }
    } else if (left_son_ptr != nullptr){
        root->TreeSize = left_son_ptr->TreeSize + 1;
        root->invalid_point_num = left_son_ptr->invalid_point_num + (root->point_deleted?1:0);
        root->down_del_num = left_son_ptr->down_del_num + (root->point_downsample_deleted?1:0);
        root->tree_downsample_deleted = left_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = left_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!left_son_ptr->tree_deleted && !root->point_deleted)){
            tmp_range_x[0] = min(left_son_ptr->node_range_x[0],root->point.x);
            tmp_range_x[1] = max(left_son_ptr->node_range_x[1],root->point.x);
            tmp_range_y[0] = min(left_son_ptr->node_range_y[0],root->point.y);
            tmp_range_y[1] = max(left_son_ptr->node_range_y[1],root->point.y); 
            tmp_range_z[0] = min(left_son_ptr->node_range_z[0],root->point.z);
            tmp_range_z[1] = max(left_son_ptr->node_range_z[1],root->point.z);  
        } else {
            if (!left_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], left_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], left_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], left_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], left_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], left_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], left_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }            
        }

    } else if (right_son_ptr != nullptr){
        root->TreeSize = right_son_ptr->TreeSize + 1;
        root->invalid_point_num = right_son_ptr->invalid_point_num + (root->point_deleted? 1:0);
        root->down_del_num = right_son_ptr->down_del_num + (root->point_downsample_deleted? 1:0);        
        root->tree_downsample_deleted = right_son_ptr->tree_downsample_deleted & root->point_downsample_deleted;
        root->tree_deleted = right_son_ptr->tree_deleted && root->point_deleted;
        if (root->tree_deleted || (!right_son_ptr->tree_deleted && !root->point_deleted)){
            tmp_range_x[0] = min(right_son_ptr->node_range_x[0],root->point.x);
            tmp_range_x[1] = max(right_son_ptr->node_range_x[1],root->point.x);
            tmp_range_y[0] = min(right_son_ptr->node_range_y[0],root->point.y);
            tmp_range_y[1] = max(right_son_ptr->node_range_y[1],root->point.y); 
            tmp_range_z[0] = min(right_son_ptr->node_range_z[0],root->point.z);
            tmp_range_z[1] = max(right_son_ptr->node_range_z[1],root->point.z); 
        } else {
            if (!right_son_ptr->tree_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], right_son_ptr->node_range_x[0]);
                tmp_range_x[1] = max(tmp_range_x[1], right_son_ptr->node_range_x[1]);
                tmp_range_y[0] = min(tmp_range_y[0], right_son_ptr->node_range_y[0]);
                tmp_range_y[1] = max(tmp_range_y[1], right_son_ptr->node_range_y[1]);
                tmp_range_z[0] = min(tmp_range_z[0], right_son_ptr->node_range_z[0]);
                tmp_range_z[1] = max(tmp_range_z[1], right_son_ptr->node_range_z[1]);                
            }
            if (!root->point_deleted){
                tmp_range_x[0] = min(tmp_range_x[0], root->point.x);
                tmp_range_x[1] = max(tmp_range_x[1], root->point.x);
                tmp_range_y[0] = min(tmp_range_y[0], root->point.y);
                tmp_range_y[1] = max(tmp_range_y[1], root->point.y);
                tmp_range_z[0] = min(tmp_range_z[0], root->point.z);
                tmp_range_z[1] = max(tmp_range_z[1], root->point.z);                 
            }            
        }
    } else {
        root->TreeSize = 1;
        root->invalid_point_num = (root->point_deleted? 1:0);
        root->down_del_num = (root->point_downsample_deleted? 1:0);
        root->tree_downsample_deleted = root->point_downsample_deleted;
        root->tree_deleted = root->point_deleted;
        tmp_range_x[0] = root->point.x;
        tmp_range_x[1] = root->point.x;        
        tmp_range_y[0] = root->point.y;
        tmp_range_y[1] = root->point.y; 
        tmp_range_z[0] = root->point.z;
        tmp_range_z[1] = root->point.z;                 
    }
    memcpy(root->node_range_x,tmp_range_x,sizeof(tmp_range_x));
    memcpy(root->node_range_y,tmp_range_y,sizeof(tmp_range_y));
    memcpy(root->node_range_z,tmp_range_z,sizeof(tmp_range_z));
    float x_L = (root->node_range_x[1] - root->node_range_x[0]) * 0.5;
    float y_L = (root->node_range_y[1] - root->node_range_y[0]) * 0.5;
    float z_L = (root->node_range_z[1] - root->node_range_z[0]) * 0.5;
    root->radius_sq = x_L*x_L + y_L * y_L + z_L * z_L;    
    if (left_son_ptr != nullptr) left_son_ptr -> father_ptr = root;
    if (right_son_ptr != nullptr) right_son_ptr -> father_ptr = root;
    if (root == Root_Node && root->TreeSize > 3){
        KD_TREE_NODE * son_ptr = root->left_son_ptr;
        if (son_ptr == nullptr) son_ptr = root->right_son_ptr;
        float tmp_bal = float(son_ptr->TreeSize) / (root->TreeSize-1);
        root->alpha_del = float(root->invalid_point_num)/ root->TreeSize;
        root->alpha_bal = (tmp_bal>=0.5-EPSS)?tmp_bal:1-tmp_bal;
    }   
    return;
}

/**
 * @brief 将KD树扁平化，将节点中的点存储到指定的容器中，并根据存储类型处理被删除的点
 * 
 * @param root KD树的根节点指针，用于遍历整棵树
 * @param Storage 用于存储未被删除点的向量容器
 * @param storage_type 删除点的存储类型，决定如何处理被删除的点
 * 
 * @note 该函数通过递归遍历KD树，将未被删除的点添加到Storage中，
 *       并根据storage_type参数决定是否以及如何存储被删除的点
 */
template <typename PointType>
void KD_TREE<PointType>::flatten(KD_TREE_NODE * root, PointVector &Storage, delete_point_storage_set storage_type){
    // 递归终止条件：节点为空则直接返回
    if (root == nullptr) return;
    
    // 下推操作，处理节点的延迟删除标记
    Push_Down(root);
    
    // 如果当前节点的点未被删除，则将其添加到存储容器中
    if (!root->point_deleted) {
        Storage.push_back(root->point);
    }
    
    // 递归处理左子树和右子树
    flatten(root->left_son_ptr, Storage, storage_type);
    flatten(root->right_son_ptr, Storage, storage_type);
    
    // 根据存储类型处理被删除的点
    switch (storage_type)
    {
    case NOT_RECORD:
        // 不记录被删除的点，直接跳出
        break;
    case DELETE_POINTS_REC:
        // 记录被删除但未被下采样删除的点到Points_deleted容器中
        if (root->point_deleted && !root->point_downsample_deleted) {
            Points_deleted.push_back(root->point);
        }       
        break;
    case MULTI_THREAD_REC:
        // 在多线程环境下记录被删除但未被下采样删除的点到Multithread_Points_deleted容器中
        if (root->point_deleted  && !root->point_downsample_deleted) {
            Multithread_Points_deleted.push_back(root->point);
        }
        break;
    default:
        break;
    }     
    return;
}

template <typename PointType>
void KD_TREE<PointType>::delete_tree_nodes(KD_TREE_NODE ** root){ 
    if (*root == nullptr) return;
    Push_Down(*root);    
    delete_tree_nodes(&(*root)->left_son_ptr);
    delete_tree_nodes(&(*root)->right_son_ptr);
    
    pthread_mutex_destroy( &(*root)->push_down_mutex_lock);         
    delete *root;
    *root = nullptr;                    

    return;
}

template <typename PointType>
bool KD_TREE<PointType>::same_point(PointType a, PointType b){
    return (fabs(a.x-b.x) < EPSS && fabs(a.y-b.y) < EPSS && fabs(a.z-b.z) < EPSS );
}

template <typename PointType>
float KD_TREE<PointType>::calc_dist(PointType a, PointType b){
    float dist = 0.0f;
    dist = (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z);
    return dist;
}

template <typename PointType>
float KD_TREE<PointType>::calc_box_dist(KD_TREE_NODE * node, PointType point){
    if (node == nullptr) return INFINITY;
    float min_dist = 0.0;
    if (point.x < node->node_range_x[0]) min_dist += (point.x - node->node_range_x[0])*(point.x - node->node_range_x[0]);
    if (point.x > node->node_range_x[1]) min_dist += (point.x - node->node_range_x[1])*(point.x - node->node_range_x[1]);
    if (point.y < node->node_range_y[0]) min_dist += (point.y - node->node_range_y[0])*(point.y - node->node_range_y[0]);
    if (point.y > node->node_range_y[1]) min_dist += (point.y - node->node_range_y[1])*(point.y - node->node_range_y[1]);
    if (point.z < node->node_range_z[0]) min_dist += (point.z - node->node_range_z[0])*(point.z - node->node_range_z[0]);
    if (point.z > node->node_range_z[1]) min_dist += (point.z - node->node_range_z[1])*(point.z - node->node_range_z[1]);
    return min_dist;
}

template <typename PointType> bool KD_TREE<PointType>::point_cmp_x(PointType a, PointType b) { return a.x < b.x;}
template <typename PointType> bool KD_TREE<PointType>::point_cmp_y(PointType a, PointType b) { return a.y < b.y;}
template <typename PointType> bool KD_TREE<PointType>::point_cmp_z(PointType a, PointType b) { return a.z < b.z;}

// manual queue
template <typename T>
void MANUAL_Q<T>::clear(){
    head = 0;
    tail = 0;
    counter = 0;
    is_empty = true;
    return;
}

template <typename T>
void MANUAL_Q<T>::pop(){
    if (counter == 0) return;
    head ++;
    head %= Q_LEN;
    counter --;
    if (counter == 0) is_empty = true;
    return;
}

template <typename T>
T MANUAL_Q<T>::front(){
    return q[head];
}

template <typename T>
T MANUAL_Q<T>::back(){
    return q[tail];
}

template <typename T>
void MANUAL_Q<T>::push(T op){
    q[tail] = op;
    counter ++;
    if (is_empty) is_empty = false;
    tail ++;
    tail %= Q_LEN;
}

template <typename T>
bool MANUAL_Q<T>::empty(){
    return is_empty;
}

template <typename T>
int MANUAL_Q<T>::size(){
    return counter;
}

template class KD_TREE<ikdTree_PointType>;
template class KD_TREE<pcl::PointXYZ>;
template class KD_TREE<pcl::PointXYZI>;
template class KD_TREE<pcl::PointXYZINormal>;