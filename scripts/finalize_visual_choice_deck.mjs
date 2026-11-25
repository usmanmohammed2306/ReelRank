// Validate the editable chart, slide geometry and font family against the supplied ASU template.
import path from 'node:path';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';

const [candidate, final, python, skillDirectory, workspace, template] = process.argv.slice(2);
if(!template) throw Error('Supply candidate, fresh final path, Python, presentation skill, workspace and official ASU template');
const validators=path.join(skillDirectory,'container_tools');
const {finalizePresentation}=await import(pathToFileURL(path.join(validators,'artifact_tool_utils.mjs')).href);
await finalizePresentation({
  workspaceDir:path.resolve(workspace),candidatePath:path.resolve(candidate),finalPath:path.resolve(final),
  pythonExecutable:python,integrityValidatorPath:path.join(validators,'inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(validators,'inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','9144000,5143500','--validate-heading-fit'],
  explicitTotalSlideCount:11,requiredNativeChartOwnerSlides:[],requiredNativeTableOwnerSlides:[],
  materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
  fontPolicy:{basis:'reference',families:['Arial'],referencePath:path.resolve(template),referenceSha256:createHash('sha256').update(fs.readFileSync(template)).digest('hex')},
  receiptPath:path.resolve(candidate+'.validation.json')
});
