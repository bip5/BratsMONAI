from monai.data import decollate_batch



def validate(val_loader, epoch, best_metric, best_metric_epoch, sheet_name=None, save_name=None, custom_inference=None):
    def default_inference(val_data):
        val_inputs = val_data["image"].to(device)
        val_masks = val_data["mask"].to(device)
        
        val_data["pred"] = inference(val_inputs, model)
        # print(val_data['pred'][0].shape,"val_data['pred'][0].shape)")
        val_data = [post_trans(i) for i in decollate_batch(val_data)]
        val_outputs, val_masks = from_engine(["pred", "mask"])(val_data)

        
        return val_outputs, val_masks
    
    model.eval()
    alt_metrics = []
   
    
    #picks default when custom is none
    run_inference = custom_inference or default_inference

    
    with torch.no_grad():
        for val_data in val_loader:
            val_outputs, val_masks = run_inference(val_data)
            
            # Consistent dice calculation
            val_outputs = [tensor.to(device) for tensor in val_outputs]
            val_masks = [tensor.to(device) for tensor in val_masks]
            #need to invert image as well for plotting purposes
            inverter = transforms.Invertd(keys="image", transform=val_transform_isles, orig_keys="image", meta_keys="image_meta_dict", nearest_interp=False, to_tensor=True)
            
            val_inputs = [inverter(x)["image"] for x in decollate_batch(val_data)]
            
            output_dir = os.path.join(output_path, job_id)
            os.makedirs(output_dir, exist_ok=True)
            
            # plot_zero(val_inputs,val_outputs,val_masks,output_dir,job_id,'001')
            
            dice_metric(y_pred=val_outputs, y=val_masks)
            dice_metric_batch(y_pred=val_outputs, y=val_masks)
            
            if loss_type == 'EdgyDice':            
                alt_metric = LesionWiseDice(val_outputs, val_masks)
                alt_metrics.append(alt_metric)
    
    if loss_type == 'EdgyDice':            
        metric = np.mean(alt_metrics)
    else:
        metric = dice_metric.aggregate().item()
    
    modes = ['isles', 'atlas']
    if training_mode not in modes:      
        metric_batch = dice_metric_batch.aggregate()
        metric_tc = metric_batch[0].item()
        metric_wt = metric_batch[1].item()
        metric_et = metric_batch[2].item()
    
    dice_metric.reset()
    dice_metric_batch.reset()

    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch + 1
        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimiser.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
        }
        
        if training_mode == 'CV_fold':   
            save_name = f"{model_name}CV{fold_num}_j{job_id}{'ep'+str(epoch+1) if checkpoint_snaps else ''}_ts{temporal_split}"
        elif sheet_name is not None:  
            save_name = f"{sheet_name}_{load_save}_j{job_id}{'_e'+str(best_metric_epoch) if checkpoint_snaps else ''}"
        else:
            save_name = f"{date.today().isoformat()}{model_name}_j{job_id}_ts{temporal_split}"
        
        saved_model = os.path.join(save_dir, save_name)
        torch.save(state, saved_model)
        
        save_name_sd=date.today().isoformat()+model_name+'_j'+str(job_id)+'_ts'+str(temporal_split)+ '_sd'
        saved_model_sd=os.path.join(save_dir, save_name_sd)
        torch.save(
            model.state_dict(),
            saved_model_sd,
        )
        print(f"Saved new best metric model: {saved_model}")
    
    if training_mode in modes:
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"            
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    else:
        print(
            f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            f"\nbest mean dice: {best_metric:.4f}"
            f" at epoch: {best_metric_epoch}"
        )
    
    return save_name, best_metric, best_metric_epoch, metric
