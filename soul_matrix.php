<?php

/**
 * Soul Matrix: the persona-vector core as one dependency-free PHP file (PHP 8.1+).
 *
 *   seed ──sha256──▶ z ∈ ℝ¹⁷ (independent, clipped normals)
 *        ──L (Cholesky of the axis correlation matrix R)──▶ x = A·z + b ∈ ℝ¹⁷
 *   clone: given a few observed axes of x (with noise), draw z* from the exact
 *          Gaussian posterior (Matheron's rule) so the unobserved axes move with R.
 *
 * Same seed → same person on every machine: all randomness comes from sha256, never mt_rand.
 * Checked against the private engine (verify/): z and conditioning identical, x within 1e-15 given the
 * engine's shift magnitudes. Archetype signs and shift magnitudes use this file's own salts.
 */

declare(strict_types=1);

final class SoulMatrix
{
    /** Axis order is fixed and append-only: Cholesky is prefix-stable, so adding an axis never changes existing people. */
    public const AXES = ['E', 'O', 'A', 'C', 'N', 'humor', 'slang', 'emoji', 'punct_heavy', 'length', 'english', 'typo', 'particle', 'activity', 'disagree', 'exclaim', 'swear'];

    /** Non-zero off-diagonal entries of the target correlation matrix R (all other pairs are 0). */
    public const R_PAIRS = [
        ['A', 'C', 0.20], ['N', 'C', -0.25], ['N', 'A', -0.20], ['E', 'O', 0.20],
        ['E', 'emoji', 0.25], ['E', 'activity', 0.30], ['E', 'exclaim', 0.25],
        ['O', 'english', 0.15], ['O', 'length', 0.20],
        ['A', 'disagree', -0.40], ['A', 'swear', -0.30], ['C', 'swear', -0.20],
        ['C', 'punct_heavy', -0.10], ['N', 'punct_heavy', 0.25], ['C', 'typo', -0.35],
        ['humor', 'slang', 0.30], ['slang', 'emoji', 0.30], ['slang', 'particle', 0.35],
    ];

    /**
     * Optional coarse archetype: one sign per axis below. x_P = r·s_P·d_P + √(1−r²)·(L·z)_P,
     * d_P = |normal|. R is written in z-space; cells touching a shifted axis are divided by √(1−r²)
     * so that, averaged over all sign patterns, Corr(x) returns to the target R.
     */
    public const SHIFT = ['E' => 0.70, 'O' => 0.70, 'A' => 0.45, 'C' => 0.50];

    public const Z_CLIP = 2.5;

    /** 97.5% quantile of χ²₁₇: people with ‖z‖² above this are rejected as caricatures. */
    public const CHI2_975 = 30.19;

    /** Variance of a standard normal clipped at ±2.5. */
    public const PRIOR_VAR = 0.97756;

    private const TWO_POW_52 = 4503599627370496;

    private const A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];

    private const B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];

    private const C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];

    private const D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];

    private static ?array $l = null;

    // ------------------------------------------------------------ deterministic randomness

    /** Uniform in the open interval (0,1) from the first 52 bits of sha256. */
    public static function unit(string ...$parts): float
    {
        $k = hexdec(substr(hash('sha256', implode('|', $parts)), 0, 13));

        return ($k + 0.5) / self::TWO_POW_52;
    }

    /** Standard normal, rounded to 6 places (absorbs last-ulp libm differences), clipped at ±2.5. */
    public static function normal(string ...$parts): float
    {
        $z = round(self::invNormal(self::unit(...$parts)), 6);

        return max(-self::Z_CLIP, min(self::Z_CLIP, $z));
    }

    /** Φ⁻¹(u), Acklam's rational approximation (relative error ≈ 1.15e-9). */
    public static function invNormal(float $u): float
    {
        if (! ($u > 0.0 && $u < 1.0)) {
            throw new InvalidArgumentException("invNormal needs 0 < u < 1, got {$u}");
        }
        [$a, $b, $c, $d] = [self::A, self::B, self::C, self::D];
        if ($u < 0.02425 || $u > 1 - 0.02425) {
            $q = sqrt(-2 * log($u < 0.02425 ? $u : 1 - $u));
            $x = ((((($c[0] * $q + $c[1]) * $q + $c[2]) * $q + $c[3]) * $q + $c[4]) * $q + $c[5])
                / (((($d[0] * $q + $d[1]) * $q + $d[2]) * $q + $d[3]) * $q + 1);

            return $u < 0.02425 ? $x : -$x;
        }
        $q = $u - 0.5;
        $r = $q * $q;

        return ((((($a[0] * $r + $a[1]) * $r + $a[2]) * $r + $a[3]) * $r + $a[4]) * $r + $a[5]) * $q
            / ((((($b[0] * $r + $b[1]) * $r + $b[2]) * $r + $b[3]) * $r + $b[4]) * $r + 1);
    }

    // ------------------------------------------------------------ the matrix

    /** @return list<list<float>> target R, or the z-space R adjusted for the archetype shift */
    public static function correlation(bool $adjusted = true): array
    {
        $index = array_flip(self::AXES);
        $n = count(self::AXES);
        $m = array_fill(0, $n, array_fill(0, $n, 0.0));
        for ($i = 0; $i < $n; $i++) {
            $m[$i][$i] = 1.0;
        }
        foreach (self::R_PAIRS as [$p, $q, $r]) {
            if ($adjusted) {
                foreach ([$p, $q] as $axis) {
                    if (isset(self::SHIFT[$axis])) {
                        $r /= sqrt(1 - self::SHIFT[$axis] ** 2);
                    }
                }
            }
            $m[$index[$p]][$index[$q]] = $m[$index[$q]][$index[$p]] = $r;
        }

        return $m;
    }

    /** @param list<list<float>> $m @return list<list<float>> lower-triangular L with L·Lᵀ = m */
    public static function cholesky(array $m): array
    {
        $n = count($m);
        $l = array_fill(0, $n, array_fill(0, $n, 0.0));
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j <= $i; $j++) {
                $sum = (float) $m[$i][$j];
                for ($k = 0; $k < $j; $k++) {
                    $sum -= $l[$i][$k] * $l[$j][$k];
                }
                if ($i === $j) {
                    if ($sum <= 1e-12) {
                        throw new RuntimeException("matrix is not positive definite at pivot {$i}");
                    }
                    $l[$i][$i] = sqrt($sum);
                } else {
                    $l[$i][$j] = $sum / $l[$j][$j];
                }
            }
        }

        return $l;
    }

    public static function L(): array
    {
        return self::$l ??= self::cholesky(self::correlation(true));
    }

    // ------------------------------------------------------------ a person

    /** @return array<string, float> axis => z */
    public static function z(int $seed): array
    {
        $z = [];
        foreach (self::AXES as $axis) {
            $z[$axis] = self::normal('pf', '1', (string) $seed, $axis);
        }

        return $z;
    }

    /** @return list<float> shift magnitudes d_P for E, O, A, C */
    public static function shiftMagnitudes(int $seed): array
    {
        return array_values(array_map(fn (string $axis) => abs(self::normal('soul-matrix', '1', (string) $seed, 'shift_'.$axis)), array_keys(self::SHIFT)));
    }

    /** @return array<string, int> E/O/A/C => ±1, from a string like "+-+-" or drawn from the seed */
    public static function signs(int $seed, ?string $pattern = null): array
    {
        $signs = [];
        foreach (array_keys(self::SHIFT) as $i => $axis) {
            $signs[$axis] = $pattern !== null
                ? ($pattern[$i] === '+' ? 1 : -1)
                : (self::unit('soul-matrix', '1', (string) $seed, 'sign_'.$axis) < 0.5 ? -1 : 1);
        }

        return $signs;
    }

    /**
     * x = A·z + b: row i of A is row i of L (times √(1−r²) on shifted axes), b_P = r·s_P·d_P.
     *
     * @return array{0: list<list<float>>, 1: list<float>}
     */
    public static function design(array $mag, array $signs): array
    {
        $l = self::L();
        $n = count(self::AXES);
        $magBy = array_combine(array_keys(self::SHIFT), $mag);
        $a = [];
        $b = [];
        foreach (self::AXES as $i => $axis) {
            $row = array_fill(0, $n, 0.0);
            for ($j = 0; $j <= $i; $j++) {
                $row[$j] = (float) $l[$i][$j];
            }
            $bi = 0.0;
            if (isset(self::SHIFT[$axis])) {
                $r = self::SHIFT[$axis];
                $row = array_map(fn (float $v) => $v * sqrt(1 - $r * $r), $row);
                $bi = $r * $signs[$axis] * $magBy[$axis];
            }
            $a[$i] = $row;
            $b[$i] = $bi;
        }

        return [$a, $b];
    }

    /** @return array<string, float> */
    public static function x(array $z, array $mag, array $signs): array
    {
        [$a, $b] = self::design($mag, $signs);
        $zv = array_map(fn (string $axis) => $z[$axis], self::AXES);
        $x = [];
        foreach (self::AXES as $i => $axis) {
            $x[$axis] = self::dot($a[$i], $zv) + $b[$i];
        }

        return $x;
    }

    public static function inBounds(array $e): bool
    {
        $norm = 0.0;
        foreach ($e as $v) {
            if (abs($v) > self::Z_CLIP) {
                return false;
            }
            $norm += $v * $v;
        }

        return $norm <= self::CHI2_975;
    }

    // ------------------------------------------------------------ clone, distance, blend

    /**
     * Clone: condition a seed's prior z on observed axes (Matheron's rule, with observation noise).
     *   z* = z₀ + A_oᵀ (A_o A_oᵀ + Σ)⁻¹ (x_o − A_o z₀ − b_o − ε),   ε ~ N(0, Σ) drawn from the seed.
     * This is an exact posterior sample of z | (A_o z + b_o + noise = x_o). var → 0 pins an axis;
     * a large var falls back to the prior. Unobserved axes move through R. Out of bounds → null.
     *
     * @param  array<string, array{x: float, var: float}>  $obs
     * @return array<string, float>|null
     */
    public static function condition(array $z0, array $mag, array $signs, array $obs, int $seed): ?array
    {
        if ($obs === []) {
            return self::inBounds($z0) ? $z0 : null;
        }
        $index = array_flip(self::AXES);
        [$a, $b] = self::design($mag, $signs);
        $ev = array_map(fn (string $axis) => (float) $z0[$axis], self::AXES);
        $rows = array_map(function (string $axis) use ($index) {
            return $index[$axis] ?? throw new InvalidArgumentException("unknown axis: {$axis}");
        }, array_keys($obs));
        $k = count($rows);
        $s = [];
        $res = [];
        foreach ($rows as $p => $i) {
            for ($q = 0; $q < $k; $q++) {
                $s[$p][$q] = self::dot($a[$i], $a[$rows[$q]]);
            }
            $var = (float) $obs[self::AXES[$i]]['var'];
            $s[$p][$p] += $var / self::PRIOR_VAR;
            $eps = $var > 0 ? sqrt($var / self::PRIOR_VAR) * self::normal('soul', '1', (string) $seed, 'eps', self::AXES[$i]) : 0.0;
            $res[$p] = (float) $obs[self::AXES[$i]]['x'] - self::dot($a[$i], $ev) - $b[$i] - $eps;
        }
        $w = self::solve($s, $res);
        $out = [];
        foreach (self::AXES as $j => $axis) {
            $d = 0.0;
            foreach ($rows as $p => $i) {
                $d += $a[$i][$j] * $w[$p];
            }
            $out[$axis] = round($ev[$j] + $d, 6);
        }

        return self::inBounds($out) ? $out : null;
    }

    public static function distance(array $za, array $zb): float
    {
        $sum = 0.0;
        foreach ($za as $k => $v) {
            $sum += ($v - $zb[$k]) ** 2;
        }

        return sqrt($sum);
    }

    /** z_t = ((1−t)z_A + t·z_B) / √((1−t)² + t²): without the divisor a 50/50 blend has sd 0.707 and reads as bland. */
    public static function blend(array $za, array $zb, float $t): array
    {
        $norm = sqrt((1 - $t) ** 2 + $t ** 2);
        $out = [];
        foreach ($za as $k => $v) {
            $out[$k] = round(max(-self::Z_CLIP, min(self::Z_CLIP, ((1 - $t) * $v + $t * $zb[$k]) / $norm)), 6);
        }

        return $out;
    }

    public static function level(float $x): string
    {
        return match (true) {
            $x < -1.2815515655446004 => 'very low',
            $x < -0.5244005127080407 => 'low',
            $x < 0.5244005127080407 => 'mid',
            $x < 1.2815515655446004 => 'high',
            default => 'very high',
        };
    }

    private static function dot(array $u, array $v): float
    {
        $sum = 0.0;
        foreach ($u as $i => $value) {
            $sum += $value * $v[$i];
        }

        return $sum;
    }

    /** Gaussian elimination with partial pivoting; A_o A_oᵀ + Σ is always invertible because L has a positive diagonal. */
    private static function solve(array $m, array $rhs): array
    {
        $n = count($rhs);
        for ($c = 0; $c < $n; $c++) {
            $pivot = $c;
            for ($r = $c + 1; $r < $n; $r++) {
                if (abs($m[$r][$c]) > abs($m[$pivot][$c])) {
                    $pivot = $r;
                }
            }
            [$m[$c], $m[$pivot]] = [$m[$pivot], $m[$c]];
            [$rhs[$c], $rhs[$pivot]] = [$rhs[$pivot], $rhs[$c]];
            for ($r = $c + 1; $r < $n; $r++) {
                $f = $m[$r][$c] / $m[$c][$c];
                for ($k = $c; $k < $n; $k++) {
                    $m[$r][$k] -= $f * $m[$c][$k];
                }
                $rhs[$r] -= $f * $rhs[$c];
            }
        }
        $x = array_fill(0, $n, 0.0);
        for ($r = $n - 1; $r >= 0; $r--) {
            $s = $rhs[$r];
            for ($k = $r + 1; $k < $n; $k++) {
                $s -= $m[$r][$k] * $x[$k];
            }
            $x[$r] = $s / $m[$r][$r];
        }

        return $x;
    }
}

// ------------------------------------------------------------ CLI

if (PHP_SAPI === 'cli' && realpath($argv[0] ?? '') === __FILE__) {
    $cmd = $argv[1] ?? 'help';
    $person = function (int $seed, ?string $pattern) {
        $z = SoulMatrix::z($seed);
        $mag = SoulMatrix::shiftMagnitudes($seed);
        $signs = SoulMatrix::signs($seed, $pattern);

        return [$z, $mag, $signs, SoulMatrix::x($z, $mag, $signs)];
    };
    $round = fn (array $v) => array_map(fn (float $f) => round($f, 3), $v);
    $out = match ($cmd) {
        'forge' => (function () use ($argv, $person, $round) {
            $seed = (int) ($argv[2] ?? 0);
            [$z, , $signs, $x] = $person($seed, $argv[3] ?? null);

            return ['seed' => $seed, 'signs' => $signs, 'in_bounds' => SoulMatrix::inBounds($z), 'z' => $round($z), 'x' => $round($x), 'level' => array_map([SoulMatrix::class, 'level'], $x)];
        })(),
        'clone' => (function () use ($argv, $person, $round) {
            $seed = (int) ($argv[2] ?? 0);
            $obs = [];
            foreach (array_slice($argv, 3) as $arg) {
                [$axis, $value] = explode('=', $arg, 2);
                [$xv, $var] = array_pad(explode(':', $value, 2), 2, '0.05');
                $obs[$axis] = ['x' => (float) $xv, 'var' => (float) $var];
            }
            [$z, $mag, $signs, $x] = $person($seed, null);
            $zc = SoulMatrix::condition($z, $mag, $signs, $obs, $seed);
            if ($zc === null) {
                return ['seed' => $seed, 'error' => 'conditioned person falls outside the χ² bound; soften the observations or raise var'];
            }
            $xc = SoulMatrix::x($zc, $mag, $signs);

            $moved = array_combine(array_keys($x), array_map(fn ($a, $b) => $b - $a, $x, $xc));

            return ['seed' => $seed, 'observed' => $obs, 'before' => $round($x), 'after' => $round($xc), 'moved' => $round($moved)];
        })(),
        'blend' => (function () use ($argv, $round) {
            [$a, $b, $t] = [(int) ($argv[2] ?? 0), (int) ($argv[3] ?? 1), (float) ($argv[4] ?? 0.5)];

            return ['a' => $a, 'b' => $b, 't' => $t, 'z' => $round(SoulMatrix::blend(SoulMatrix::z($a), SoulMatrix::z($b), $t)), 'distance_ab' => round(SoulMatrix::distance(SoulMatrix::z($a), SoulMatrix::z($b)), 3)];
        })(),
        default => "usage:\n  php soul_matrix.php forge <seed> [+-+-]\n  php soul_matrix.php clone <seed> axis=value[:var] ...\n  php soul_matrix.php blend <seedA> <seedB> <t>\n",
    };
    echo is_string($out) ? $out : json_encode($out, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE)."\n";
}
