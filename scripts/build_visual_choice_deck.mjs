// Build the editable ASU presentation from its original master and the verified experiment.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';

// Ordinary slide elements keep the results visible in viewers without chart support.
export function replacePublishedMethodsChart(slide, rows) {
  for (const chart of [...slide.charts.items]) slide.charts.deleteById(chart.id);
  for (const shape of [...slide.shapes.items]) {
    if (shape.name.startsWith('Results chart ')) shape.delete();
  }
  const barLeft = 166, plotWidth = 394, plotTop = 184, rowHeight = 35;
  const scale = value => value * plotWidth / 3500;
  const add = (name, left, top, width, height, fill = 'none', label = '', opt = {}) => {
    const element = slide.shapes.add({
      name: 'Results chart ' + name, geometry: label ? 'textbox' : 'rect',
      position: {left, top, width, height}, fill, line: {fill: 'none', width: 0}
    });
    if (label) {
      element.text = label;
      element.text.style = {
        typeface: 'Arial', fontSize: opt.fontSize || 15, color: opt.color || '#191919',
        alignment: opt.align || 'left', bold: !!opt.bold, verticalAlignment: 'middle',
        wrap: 'none', insets: {left: 0, right: 0, top: 0, bottom: 0}
      };
    }
  };
  for (const tick of [0, 1000, 2000, 3000]) {
    const x = barLeft + scale(tick);
    add('grid ' + tick, x, plotTop - 3, 0.8, rows.length * rowHeight, '#E4DEDD');
    add('tick ' + tick, x - 29, plotTop + rows.length * rowHeight - 2, 58, 20,
      'none', tick.toLocaleString('en-US'), {align: 'center', color: '#54515A', fontSize: 13});
  }
  add('axis', barLeft, plotTop + rows.length * rowHeight - 3, plotWidth, 0.8, '#D7D2D0');
  rows.forEach(([label, value], index) => {
    const y = plotTop + index * rowHeight;
    const width = scale(value);
    add(label + ' name', 42, y, 115, 27, 'none', label, {align: 'right'});
    add(label + ' bar', barLeft, y, width, 27, '#8C1D40');
    add(label + ' count', barLeft + width + 6, y, 56, 27, 'none',
      value.toLocaleString('en-US'), {bold: true});
  });
}

export async function applyVisualChoiceDeck(deck, inputs) {
const {audit,ltr,news}=inputs;
const confirmed=audit.confirmation;
const originalSlides=[...deck.slides.items];
let S;
if(originalSlides.length===19){
  const originalPages=[8,12,14,13,13,16,12,13,16,13,11];
  S=originalPages.map(index=>deck.slides.add({layoutId:originalSlides[index-1].layout.id}));
  deck.slides.delete(originalSlides.map((slide,i)=>i));
}else if(originalSlides.length===11){
  S=originalSlides;
}else throw new Error('Use the official 2026 template or the existing eleven-page course presentation');
for(const slide of S){
  slide.speakerNotes.text='';
  for(const sh of [...slide.shapes.items]) sh.delete();
  for(const image of [...slide.images.items]) image.delete();
  for(const chart of [...slide.charts.items]) slide.charts.deleteById(chart.id);
}
const newKey='personal50', h=confirmed.hits, m=confirmed.metrics;
const count=x=>x.toLocaleString('en-US'),delta=x=>(100*x).toFixed(2),pct=x=>delta(x)+'%';
const positionsByModel=audit.confirmation.by_original_position;
const example=audit.confirmation.kaguya_matching_lists[0];
const old=h.user_knn, cur=h[newKey], net=cur-old;
const allExternal=['popularity','ease','user_knn','implicit_mf','std_vae','mlt_vae'];
const labels={popularity:'Popularity',ease:'EASE',user_knn:'UserKNN',implicit_mf:'ImplicitMF',std_vae:'Standard VAE',mlt_vae:'Multinomial VAE',personal50:'ReelRank'};
if (confirmed.lists!==5510||ltr.features!==352||confirmed.people!==349||net!==342||Math.max(...allExternal.map(k=>h[k]))!==old) throw Error('Check the release scores before publishing');
const C={ink:'#191919',maroon:'#8C1D40',dark:'#451027',gold:'#FFC627',gray:'#54515A',pale:'#F8F3EB',rose:'#F5E6EC',light:'#DFDBD7',white:'#FFFFFF'};
function shape(s,name,x,y,w,h,fill='none',stroke='none',geom='rect',lineWidth=0) {
  return s.shapes.add({name,geometry:geom,position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:lineWidth}});
}
function txt(s,str,x,y,w,h,size=20,opt={}) {
  const sh=shape(s,opt.name||str.slice(0,70),x,y,w,h,opt.fill||'none',opt.line||'none',opt.geom||'textbox',opt.lw||0);
  sh.text=str;sh.text.style={typeface:'Arial',fontSize:size,bold:!!opt.bold,color:opt.color||C.ink,
    alignment:opt.align||'left',verticalAlignment:opt.va||'middle',wrap:opt.wrap||'square',insets:{top:0,bottom:0,left:0,right:0}};
  return sh;
}
const rule=(s,x,y,w,c=C.light)=>shape(s,'Horizontal rule',x,y,w,1.4,c);
function section(s,label,title) {
  const widths={'Introduction':116,'Problem description':187,'Methodology':126,'Results':76,'Conclusions and future work':264};
  txt(s,label,42,31,widths[label],26,18,{bold:true,fill:C.gold,align:'center',geom:'rect'});
  txt(s,title,42,70,875,80,40,{bold:true,va:'top'});
}
function band(s,y,text,opt={}) {
  shape(s,'Takeaway band',42,y,876,52,opt.bg||C.pale);
  shape(s,'Emphasis rule',42,y,5,52,opt.line||C.maroon);
  txt(s,text,60,y+3,840,46,opt.size||20,{color:opt.color||C.ink,bold:!!opt.bold});
}
function tag(s,caption,x,y,w=280){txt(s,caption,x,y,w,24,16,{bold:true,color:C.maroon});}
function big(s,str,x,y,w,h=81,size=67,color=C.maroon){txt(s,str,x,y,w,h,size,{bold:true,color});}
// 1. Title and authors. The logo and school name are inherited from the ASU master.
{
 const s=S[0];
 txt(s,'ReelRank',36,144,887,106,89,{bold:true});
 txt(s,'Which title should go first?',38,247,881,66,49,{bold:true});
 txt(s,'Learning from the movies and books people were offered',37,320,702,32,20,{bold:true,fill:C.gold,geom:'rect',align:'center'});
 txt(s,'Usman Mehboob Mohammed     Moin Palekar\nOmkar Rajesh Hundekari     Aryan Tripathi\nBadri Sri Vaishnavi Yeluri     Arya Ravindra Patil',479,397,443,95,17,{align:'right',bold:true});
}
// 2. A pair of hypothetical histories makes the new signal visible.
{
 const s=S[1];section(s,'Introduction','Two thriller picks can tell different stories');
 txt(s,'Each viewer picks two thrillers from a list of ten. What else did each viewer see?',42,133,870,44,23);
 const x0=295, step=58;
 for(const [r,n,label] of [[0,10,'Ten thrillers offered'],[1,2,'Two thrillers offered']]) {
   const y=209+r*83;
   txt(s,label,42,y,246,44,22,{bold:true});
   for(let j=0;j<10;j++) {
     const picked=j<2, thriller=j<n;
     txt(s,picked?'T':thriller?'T':'•',x0+j*step,y,46,44,22,
       {align:'center',bold:true,fill:picked?C.maroon:thriller?C.rose:C.pale,color:picked?C.white:thriller?C.maroon:'#7A7771',geom:'rect'});
   }
 }
 txt(s,'T = thriller      Dark red = picked      Dot = another genre',295,353,606,36,17,{color:C.gray});
 txt(s,'Illustrative choices',43,353,239,36,16,{color:C.gray});
 band(s,416,'ReelRank put a selected title first on 342 more lists than UserKNN, the best of six published benchmark models at first position.',{size:20});
}
// 3. Use an actual title rather than synthetic interface screenshots.
{
 const s=S[2];
 txt(s,'Problem description',40,33,185,28,19,{bold:true,fill:C.gold,align:'center',geom:'rect'});
 txt(s,'One real list.\nOne selected film.',40,83,465,110,42,{bold:true,color:C.white,va:'top'});
 txt(s,'The Tale of the Princess Kaguya (2013)',41,212,454,62,23,{bold:true,color:C.gold,va:'top'});
 const suffix=n=>String(n)+(['st','nd','rd'][(n%10)-1]||'th');
 const ranks=[[suffix(example.identity),'Original display'],[suffix(example.user_knn),'UserKNN'],[suffix(example.personal50),'ReelRank']];
 for(let i=0;i<3;i++){
   const x=40+i*152;
   txt(s,ranks[i][0],x,287,144,65,50,{bold:true,color:i===2?C.gold:C.white});
   txt(s,ranks[i][1],x,353,143,43,16,{color:C.white,va:'top'});
 }
 txt(s,'A hit means the first title was one the viewer marked for consideration.',42,411,450,63,19,{color:C.white,va:'top'});
 s.images.add({...(inputs.stillPath?{path:inputs.stillPath}:{blob:inputs.still}),contentType:'image/jpeg',alt:'A scene from The Tale of the Princess Kaguya, provided by Studio Ghibli',position:{left:529,top:0,width:431,height:540},fit:'cover',crop:{left:0,top:0,right:0.5685,bottom:0}});
 txt(s,'Studio Ghibli, 2013',542,499,164,24,14,{color:C.white,fill:C.ink,geom:'rect',align:'center'});
}
// 4. Pipeline, described in three connected stages.
{
 const s=S[3];section(s,'Methodology','From one viewer’s history to a ranked list');
 const stages=[
  {x:42,heading:'Compare',one:'Titles offered',two:'Titles selected',body:'Represent each title using rating patterns, catalog tags and item features.'},
  {x:347,heading:'Estimate',one:'Selected minus offered',two:'Personal preference',body:'Estimate which traits stand out in this viewer’s picks. Regularization stabilizes short histories.'},
  {x:654,heading:'Rank',one:'Personal preference',two:'Ten candidate scores',body:'Combine the profile with catalog information and six published recommender scores.'}
 ];
 for(const [i,t] of stages.entries()){
   tag(s,t.heading,t.x,155,226);
   rule(s,t.x,193,246,C.gold);
   txt(s,t.one,t.x,210,249,33,20,{bold:true});
   txt(s,'↓',t.x,249,45,34,28,{color:C.maroon,bold:true});
   txt(s,t.two,t.x,294,251,30,20,{bold:true,color:C.maroon});
   txt(s,t.body,t.x,343,250,93,19,{va:'top',color:C.gray});
 }
 txt(s,'Only events recorded before the new list enter the viewer’s profile.',42,452,873,39,20,{bold:true});
}
// 5. Learner and evaluation protocol.
{
 const s=S[4];section(s,'Methodology','Two models give each title a final score');
 for(const [x,top,desc] of [[42,'Conditional logit','Estimates how likely each title is to be picked from this particular set of ten.'],[495,'Gradient-boosted trees','Learns combinations of signals, such as when a strong match to the viewer matters most.']]){
   shape(s,'Model accent',x,158,5,128,C.maroon);
   txt(s,top,x+18,159,400,39,24,{bold:true});
   txt(s,desc,x+18,207,386,82,21,{va:'top'});
 }
 txt(s,'50 / 50',42,311,201,64,45,{bold:true,color:C.maroon});
 txt(s,'Blend of the two standardized scores, chosen on development data.',244,314,664,60,23);
 rule(s,42,399,876);
 txt(s,count(ltr.development_people)+' development participants',42,420,409,49,22,{bold:true});
 txt(s,'→',440,422,55,45,30,{color:C.maroon,bold:true});
 txt(s,count(confirmed.people)+' separate test participants',504,420,413,49,22,{bold:true});
}
// 6. First-position accuracy versus the strongest published individual baseline.
{
 const s=S[5];section(s,'Results','A marked title came first on 2,920 lists');
 txt(s,'Correct first recommendations on the same 5,510 test lists',42,136,876,39,22);
 for(const [i,entry] of [[old,'UserKNN',pct(m.user_knn.hit1)],[ltr.confirmation.first_hits.lambdamart,'LambdaMART',pct(ltr.confirmation.metrics.lambdamart.hit1)],[cur,'ReelRank',pct(m[newKey].hit1)]].entries()){
   const x=42+298*i;rule(s,x,195,264,i===2?C.maroon:C.light);
   txt(s,entry[1],x,211,268,37,22,{bold:i===2});
   big(s,count(entry[0]),x,253,267,73,59,i===2?C.maroon:C.gray);
   txt(s,entry[2]+' of lists',x,339,267,36,19,{color:C.gray});
 }
 txt(s,'+342 vs. UserKNN (6.21 percentage points, 95% interval 4.83 to 7.55).',42,395,875,34,21,{bold:true,color:C.maroon});
 txt(s,'+38 vs. LambdaMART with the same inputs (0.69 percentage points, interval −0.36 to 1.75).',42,433,875,37,19);
}
// 7. Original externally published methods on identical held-out candidates.
{
 const s=S[6];section(s,'Results','How did the published methods compare?');
 txt(s,'Correct first recommendations out of 5,510 movie and book lists',42,132,870,38,21);
 replacePublishedMethodsChart(s, [...allExternal,newKey].sort((a,b)=>h[b]-h[a]).map(key=>[labels[key],h[key]]));
 shape(s,'Whole-list result edge',592,187,4,212,C.gold);
 big(s,m[newKey].ndcg.toFixed(4),615,203,277,78,56);
 txt(s,'ReelRank nDCG@10',617,287,279,39,23,{bold:true});
 txt(s,'ImplicitMF led the six SM-RS models at '+m.implicit_mf.ndcg.toFixed(4)+'. This metric gives more credit to choices near the top.',618,339,275,103,19,{color:C.gray,va:'top'});
}
// 8. Original positions of the titles promoted, versus UserKNN.
{
 const s=S[7];section(s,'Results','Which titles accounted for the extra hits?');
 txt(s,'Net change in correct first choices versus UserKNN, by original display position',42,136,879,52,21);
 const positions=Object.fromEntries(['first','positions_2_to_5','positions_6_to_10'].map((k,i)=>[k,positionsByModel.personal50[i]-positionsByModel.user_knn[i]]));
 for(const [i,[key,label]] of [['first','First'],['positions_2_to_5','Positions 2 to 5'],['positions_6_to_10','Positions 6 to 10']].entries()){
   const x=42+i*298, v=positions[key];
   rule(s,x,224,264,C.gold);
   big(s,(v>0?'+':'')+count(v),x,246,272,91,64,i?C.maroon:C.gray);
   txt(s,label,x,351,271,37,21,{bold:true});
 }
 band(s,410,'Of the net '+count(net)+' gains, '+count(positions.positions_2_to_5+positions.positions_6_to_10)+' came from titles originally below first place.',{size:20});
}
// 9. Chronological sensitivity using published collaborative comparisons.
{
 const s=S[8];section(s,'Results','Would the lead hold for later participants?');
 txt(s,'Correct first choices on 2,358 lists from 131 later participants.',42,137,863,53,22);
 for(const [i,k] of ['lambdamart','user_knn',newKey].entries()){
   const x=42+i*298;
   rule(s,x,235,264,i===2?C.maroon:C.light);
   txt(s,k==='lambdamart'?'LambdaMART':labels[k],x,253,270,38,21,{bold:i===2});
   big(s,count(k==='lambdamart'?ltr.forward.first_hits.lambdamart:audit.forward.hits[k]),x,302,275,82,59,i===2?C.maroon:C.gray);
 }
 band(s,405,'ReelRank gained 73 hits over LambdaMART and finished one hit ahead of UserKNN.',{size:20});
}
// 10. Direct comparisons to published neural news methods.
{
 const s=S[9];section(s,'Results','Which headline belongs first?');
 const full=news.cohorts.full, screened=news.cohorts.screened, stats=full.models;
 const names=Object.keys(stats).filter(k=>/^(NRMS|NAML|LSTUR)-(42|123|456)$/.test(k));
 const strongest=(keys,metric)=>keys.reduce((a,b)=>stats[a][metric]>=stats[b][metric]?a:b);
 const external=['NRMS','NAML','LSTUR'].map(family=>strongest(names.filter(k=>k.startsWith(family+'-')),'first_hits'));
 const best=strongest(external,'first_hits'), publicNdcg=strongest(names,'NDCG@10'), capped='Matched50', keys=[...external,capped];
 const newsGain=full.paired[capped][best], ci=newsGain.clustered_95['Hit@1'];
 const lsturCount=names.filter(k=>k.startsWith('LSTUR-')).length;
 const tested=lsturCount===3?'three seeds each of NRMS, NAML and LSTUR':'three seeds each of NRMS and NAML and '+lsturCount+' LSTUR seed'+(lsturCount===1?'':'s');
 if(full.impressions!==376471||screened.impressions!==259161||!stats[capped]||!stats.Switch)throw Error('Incomplete matched news evaluation');
 txt(s,'Same '+count(news.matched50_fit.train_impressions)+' training lists. Tested on all '+count(full.impressions)+' MIND-large validation lists.',42,137,876,40,23);
 for(const [i,key] of keys.entries()){
   const x=42+i*222, target=i===3, model=stats[key], label=target?'ReelRank':key.split('-')[0];
   rule(s,x,202,199,target?C.maroon:C.light);
   txt(s,label,x,216,201,38,23,{bold:target});
   big(s,count(model.first_hits),x,260,208,70,48,target?C.maroon:C.gray);
   txt(s,target?'50-click history':'Checkpoint '+key.split('-')[1],x,337,207,31,17,{color:C.gray});
 }
 const ensemble=stats['Neural-9'], beyondEnsemble=stats[capped].first_hits-ensemble.first_hits;
 txt(s,'+'+count(newsGain.extra_first_hits)+' vs. '+best.split('-')[0]+', the strongest individual checkpoint. Each model saw up to 50 prior clicks.',42,382,876,40,20,{bold:true,color:C.maroon});
 band(s,430,'Our equal-weight average of nine released checkpoints reached '+count(ensemble.first_hits)+'. ReelRank added '+count(beyondEnsemble)+' hits.',{size:19});
}
// 11. A succinct close and a concrete next measurement.
{
 const s=S[10];section(s,'Conclusions and future work','What should a recommender remember?');
 txt(s,'What a viewer chose',42,157,866,69,53,{bold:true});
 txt(s,'and what else they were offered.',42,227,868,66,48,{bold:true,color:C.maroon});
 rule(s,42,326,876,C.ink);
 txt(s,'What we learned',42,352,367,34,22,{bold:true});
 txt(s,'ReelRank put a marked title first on 342 more test lists than UserKNN. LambdaMART finished close.',42,397,374,86,22,{va:'top'});
 txt(s,'What to test next',482,352,405,34,22,{bold:true});
 txt(s,'Randomize the order of the same ten titles. Measure time to first choice and what people go on to watch.',482,397,423,93,21,{va:'top'});
}
return deck;
}

export async function readChoiceDeckInputs(root, {useImagePath=false}={}) {
  const files={audit:'external_baseline_audit',ltr:'lambdamart_results',news:'mind_external_baselines'};
  const inputs=Object.fromEntries(await Promise.all(Object.entries(files).map(async ([key,name])=>[key,JSON.parse(await fs.readFile(path.join(root,'results/choice_gains',name+'.json'),'utf8'))])));
  const stillPath=path.join(root,'figures/choice_pitch_kaguya.jpg');
  return {...inputs,...(useImagePath?{stillPath}:{still:await fs.readFile(stillPath)})};
}

async function main(){
  const [template, output] = process.argv.slice(2);
  if(!template || !output) throw Error('Supply the 2026 official ASU template and output PPTX path');
  const root = fileURLToPath(new URL('..',import.meta.url));
  const lib = createRequire(import.meta.url).resolve('@oai/artifact-tool', {paths:[root,path.join(root,'.reelrank-pptx-build'),...(process.env.NODE_PATH||'').split(path.delimiter)]});
  const {FileBlob,PresentationFile} = await import(pathToFileURL(lib));
  const deck = await PresentationFile.importPptx(await FileBlob.load(template));
  await applyVisualChoiceDeck(deck,await readChoiceDeckInputs(root));
  await fs.mkdir(path.dirname(path.resolve(output)),{recursive:true});
  await (await PresentationFile.exportPptx(deck)).save(output);
}
if(typeof process!=="undefined" && process.argv[1] && import.meta.url===pathToFileURL(path.resolve(process.argv[1])).href) await main();
