const path = require('path');
const sharp = require('C:/Users/ycw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const fs = require('fs');
const root = path.resolve(__dirname, '..');
(async () => {
  for (const stem of ['figure2_intervention_and_states', 'rq2_membership_changes', 'rq1_target_fully_fitting']) {
    const source = path.join(root, 'figures', stem + '.svg');
    if (fs.existsSync(source)) {
      await sharp(source, {density: 150}).png().toFile(path.join(root, 'qa', stem + '_svg.png'));
      process.stdout.write('Rendered SVG ' + stem + '\n');
    }
  }
})().catch(e => { process.stderr.write(String(e)); process.exitCode = 1; });
