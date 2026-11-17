//! Individual operations for a buffered host. These call the batch pipeline's
//! stages directly; scheduling and worker count do not change fit behavior.
use super::*;

#[derive(Deserialize)]
#[serde(tag="type",rename_all="snake_case")]
pub enum Job {
    Prepare{frame:Frame},
    Initial{frame:Frame,source:Option<Pose>,prior:Option<Pose>,static_mode:bool},
    Neighbors{frame:Frame,bank:Vec<Fit>,neighbors:Vec<Pose>,static_mode:bool},
    Orient{frame:Frame,bank:Vec<Fit>,centers:Vec<Center>,reference:Option<Reference>,fps:f64},
    Target{frames:Vec<Frame>,startups:Vec<Option<Pose>>,anchors:Vec<usize>,index:usize,fps:f64},
    Finalize{frame:Frame,source:Option<Pose>,bank:Vec<Fit>,target:Option<f64>,static_mode:bool},
}
#[derive(Serialize,Deserialize)]
pub struct Reference{pub rvec:V3,pub frame:usize}
pub fn run(g:&Geometry,job:Job)->serde_json::Value {
    match job {
        Job::Prepare{frame}=>{
            let source=startup::prepare(g,&startup::Input{width:frame.width,height:frame.height,
                observations:frame.observations.clone(),landmark_presence:frame.landmark_presence.clone()})
                .candidates.first().map(|c|c.pose.clone());
            let anchor=source.as_ref().is_some_and(|p|{
                let w=work(&frame,p,None,vec![],vec![],false);
                p.tvec[2]>g.cube_size&&depth::supported(g,&frame,&w.camera,p)
            });
            serde_json::json!({"source":source,"anchor":anchor})
        }
        Job::Initial{frame,source,prior,static_mode}=>{
            let bank=source.map(|p|search::fit_frame(g,&work(&frame,&p,prior,vec![],vec![],static_mode),&mut vec![])).unwrap_or_default();
            serde_json::json!({"bank":bank})
        }
        Job::Neighbors{frame,bank,neighbors,static_mode}=>{
            let bank=if bank.is_empty()||neighbors.is_empty(){bank}else{
                let base=pose(&bank[0]);let prior=mean_prior(g,&base,&neighbors);
                search::fit_frame(g,&work(&frame,&base,Some(prior),neighbors,bank,static_mode),&mut vec![])
            };
            serde_json::json!({"bank":bank})
        }
        Job::Orient{frame,bank,centers,reference,fps}=>{
            let bank=if bank.is_empty(){bank}else{
                let base=pose(&bank[0]);let w=work(&frame,&base,None,vec![],vec![],false);
                orientation::orient(g,&w,bank,&centers,reference.as_ref().map(|r|(rodrigues(r.rvec).0,r.frame)),fps,&frame.colors)
            };
            let reference=bank.first().map(|f|Reference{rvec:pose(f).rvec,frame:frame.frame}).or(reference);
            serde_json::json!({"bank":bank,"reference":reference})
        }
        Job::Target{frames,startups,anchors,index,fps}=>{
            let targets=depth::targets_from_anchors(&frames,&startups,&anchors,fps);
            serde_json::json!({"target":targets[index]})
        }
        Job::Finalize{frame,source,bank,target,static_mode}=>{
            let (bank,recovered)=if !static_mode {if let (Some(p),Some(target))=(source,target){depth::recover(g,&frame,&p,bank,target)}else{(bank,false)}}else{(bank,false)};
            let mut wireframe=vec![];let mut selected=None;
            if let Some(first)=bank.first(){
                let p=pose(first);let w=work(&frame,&p,None,vec![],vec![],false);
                let s=projection::surface(g,&w.camera,first.axis,first.angles,p.rvec,p.tvec);
                wireframe=s.face_polygons.iter().zip(&s.face_front).filter_map(|(p,&front)|front.then_some(*p)).collect();
                selected=Some(p);
            }
            serde_json::json!({"bank":bank,"wireframe":wireframe,"pose":selected,"depth_recovery":recovered,"stage":"cpu_pipeline",
                "bootstrap":false,"message":if recovered{"CPU pose · depth recovery"}else{"CPU pose · neighbor refinement"}})
        }
    }
}
