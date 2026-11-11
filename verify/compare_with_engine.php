<?php

/**
 * Bit-for-bit check of soul_matrix.php against the private engine it was extracted from.
 * Needs the engine checkout: ENGINE_ROOT=/path/to/engine php verify/compare_with_engine.php
 * The engine's shift magnitudes use different hash salts, so they are passed in from the engine side.
 */

declare(strict_types=1);

require __DIR__.'/../soul_matrix.php';
$root = getenv('ENGINE_ROOT') ?: exit("set ENGINE_ROOT\n");
require $root.'/vendor/autoload.php';

use App\Persona\PersonaSpace;
use App\Persona\SoulConditioner;

$space = new PersonaSpace;
$conditioner = new SoulConditioner($space);
$letters = ['E' => ['E', 'I'], 'O' => ['N', 'S'], 'A' => ['F', 'T'], 'C' => ['J', 'P']];

$n = (int) ($argv[1] ?? 1000);
$zMismatch = 0;
$xMax = 0.0;
$cMax = 0.0;
$cCompared = 0;
$cNullAgree = 0;
for ($seed = 0; $seed < $n; $seed++) {
    $zE = $space->z($seed);
    $zS = SoulMatrix::z($seed);
    if ($zE !== $zS) {
        $zMismatch++;
    }
    $signs = SoulMatrix::signs($seed);
    $type = '';
    foreach ($letters as $axis => [$plus, $minus]) {
        $type .= $signs[$axis] === 1 ? $plus : $minus;
    }
    $mag = $space->mbtiMag($seed);
    $xE = $space->x($zE, $mag, $type);
    $xS = SoulMatrix::x($zS, $mag, $signs);
    foreach ($xE as $axis => $v) {
        $xMax = max($xMax, abs($v - $xS[$axis]));
    }

    $axes = SoulMatrix::AXES;
    $obs = [];
    foreach ([0, 1, 2] as $k) {
        $axis = $axes[(int) floor(SoulMatrix::unit('verify', (string) $seed, (string) $k) * count($axes))];
        $obs[$axis] = ['x' => SoulMatrix::normal('verify-x', (string) $seed, $axis), 'var' => [0.0, 0.05, 0.5][$k]];
    }
    $cE = $conditioner->condition($zE, $mag, $type, $obs, $seed);
    $cS = SoulMatrix::condition($zS, $mag, $signs, $obs, $seed);
    if ($cE === null || $cS === null) {
        $cNullAgree += ($cE === null && $cS === null) ? 1 : 0;

        continue;
    }
    $cCompared++;
    foreach ($cE as $axis => $v) {
        $cMax = max($cMax, abs($v - $cS[$axis]));
    }
}

printf("seeds compared:                 %d\n", $n);
printf("z vectors not identical:        %d\n", $zMismatch);
printf("max |x_engine - x_standalone|:  %.3e\n", $xMax);
printf("clones compared (3 obs each):   %d (both rejected as out-of-bounds: %d, disagreements: %d)\n", $cCompared, $cNullAgree, $n - $cCompared - $cNullAgree);
printf("max |z*_engine - z*_standalone|: %.3e\n", $cMax);
