// 有限個のデータから分散共分散行列を求める．
// 確率変数は正規分布に従うものとする．

/// x: 複数の確率変数をまとめた列ベクトル
#[inline]
pub fn calc(x: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let length = x.len();
    let mut matrix = Vec::with_capacity(length);
    for i in 0..length {
        let mut line = Vec::with_capacity(length);
        for j in 0..length {
            line.push( Cov(&x[i], &x[j]) );
        }
        matrix.push(line);
    }
    matrix
}

/// 理解のために3x3の場合に限定して実装
#[inline]
pub fn calc_3x3(x: Vec<f64>, y: Vec<f64>, z: Vec<f64>) -> [[f64; 3]; 3] {
    let cov_xy = Cov(&x, &y);
    let cov_yz = Cov(&y, &z);
    let cov_xz = Cov(&x, &z);
    [
        [V(&x), cov_xy, cov_xz],
        [cov_xy, V(&y), cov_yz],
        [cov_xz, cov_yz, V(&z)],
    ]
}

/// 期待値を計算（正規分布なので相加平均と同じ）
#[inline]
#[allow(non_snake_case)]
fn E(x: &Vec<f64>) -> f64 {
    x.iter().sum::<f64>() / (x.len() as f64)
}

/// 偏差ベクトルを計算
#[inline]
#[allow(non_snake_case)]
fn Dev(x: &Vec<f64>) -> Vec<f64> {
    let e = E(&x);
    x.iter().map(|a| a - e).collect()
}

/// 共分散
#[inline]
#[allow(non_snake_case)]
fn Cov(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let dev_x = Dev(&x);
    let dev_y = Dev(&y);
    // dev_xとdev_yの各要素を掛ける
    let tmp = dev_x.iter().zip(dev_y.iter()).map(|(a, b)| a * b).collect();
    E(&tmp)
}

/// 分散
#[inline]
#[allow(non_snake_case)]
fn V(x: &Vec<f64>) -> f64 {
    let dev_x = Dev(&x);
    let tmp = dev_x.iter().map(|a| a * a).collect();
    E(&tmp)
}


#[cfg(test)]
mod tests {
    use super::*;
    
    const EPSILON: f64 = 1e-14;

    #[test]
    fn test1() {
        // 確率変数x, y
        let x = vec![40.0, 80.0, 90.0];
        let y = vec![80.0, 90.0, 100.0];

        let sigma = calc(vec![x, y]);

        let mat_true = vec![
            vec![1400.0 / 3.0, 500.0 / 3.0],
            vec![500.0  / 3.0, 200.0 / 3.0],
        ];

        for i in 0..sigma.len() {
            for j in 0..sigma.len() {
                assert!( (sigma[i][j] - mat_true[i][j]).abs() < EPSILON );
            }
        }
    }
}
