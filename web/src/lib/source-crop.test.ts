import assert from 'node:assert/strict';
import test from 'node:test';
import {FULL_CROP, boundedCrop, cropFromPoints, cropPixels, dragCrop} from './source-crop';

test('selections drawn in either direction stay inside the source', () => {
  const a = {x: 0.8, y: 0.75}, b = {x: 0.2, y: 0.25};
  assert.deepEqual(cropFromPoints(a, b), cropFromPoints(b, a));
  assert.deepEqual(cropFromPoints({x: -1, y: -2}, {x: 2, y: 3}), FULL_CROP);
  assert.deepEqual(boundedCrop({x: NaN, y: 0, width: 1, height: 1}), FULL_CROP);
});

test('moving preserves crop size and resizing cannot cross an opposite edge', () => {
  const start = {x: 0.2, y: 0.25, width: 0.5, height: 0.5};
  assert.deepEqual(dragCrop(start, 'move', 2, -2), {...start, x: 0.5, y: 0});
  const small = dragCrop(start, 'nw', 2, 2);
  assert.ok(Math.abs(small.width - 0.01) < 1e-10);
  assert.ok(Math.abs(small.height - 0.01) < 1e-10);
  assert.equal(small.x + small.width, start.x + start.width);
  assert.equal(small.y + small.height, start.y + start.height);
  assert.deepEqual(dragCrop(start, 'se', 2, 2), {x: 0.2, y: 0.25, width: 0.8, height: 0.75});
});

test('crop uses original pixel dimensions for landscape, portrait and edge rounding', () => {
  const crop = {x: 0.25, y: 0.125, width: 0.5, height: 0.75};
  assert.deepEqual(cropPixels(crop, 3840, 2160), {x: 960, y: 270, width: 1920, height: 1620});
  assert.deepEqual(cropPixels(crop, 1080, 1920), {x: 270, y: 240, width: 540, height: 1440});
  assert.deepEqual(cropPixels({x: 0.99, y: 0.99, width: 0.01, height: 0.01}, 17, 13),
    {x: 16, y: 12, width: 1, height: 1});
  assert.equal(cropPixels(FULL_CROP, 1920, 1080), undefined);
  assert.equal(cropPixels(undefined, 1920, 1080), undefined);
});
