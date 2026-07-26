use std::hash::Hash;

/// 特定旋轉/鏡像版本的棋子細胞集合
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PieceVariant {
    pub cells: Vec<(i32, i32)>,
    pub width: i32,
    pub height: i32,
}

/// 棋子唯一識別碼
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PieceId(pub usize);

/// 標準形狀（基準方向）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PieceShape {
    pub id: PieceId,
    pub name: &'static str,
    pub base: PieceVariant,
    /// 預先計算的所有不重複變體
    pub variants: Vec<PieceVariant>,
}

impl PieceVariant {
    /// 建立新的變體，自動標準化至 (0,0) 並計算寬高
    pub fn new(cells: Vec<(i32, i32)>) -> Self {
        let cells = normalize(cells);
        let width = cells.iter().map(|&(x, _)| x).max().unwrap_or(0) + 1;
        let height = cells.iter().map(|&(_, y)| y).max().unwrap_or(0) + 1;
        Self { cells, width, height }
    }

    /// 順時針旋轉 90°：(x, y) → (y, -x)
    pub fn rotated(&self) -> Self {
        let cells: Vec<(i32, i32)> = self
            .cells
            .iter()
            .map(|&(x, y)| (y, -x))
            .collect();
        Self::new(cells)
    }

    /// 水平鏡像：(x, y) → (-x, y)
    pub fn mirrored(&self) -> Self {
        let cells: Vec<(i32, i32)> = self
            .cells
            .iter()
            .map(|&(x, y)| (-x, y))
            .collect();
        Self::new(cells)
    }

    /// 取得所有不重複的旋轉變體（0°, 90°, 180°, 270°）
    pub fn all_rotations(&self) -> Vec<PieceVariant> {
        let mut variants = Vec::with_capacity(4);
        let mut current = self.clone();
        for _ in 0..4 {
            variants.push(current.clone());
            current = current.rotated();
        }
        variants.sort();
        variants.dedup();
        variants
    }
}

impl PieceShape {
    /// 從基準細胞建立 PieceShape，並預先計算所有變體
    pub fn new(id: PieceId, name: &'static str, cells: Vec<(i32, i32)>) -> Self {
        let base = PieceVariant::new(cells);
        let variants = Self::compute_all_variants(&base);
        Self { id, name, base, variants }
    }

    /// 計算 8 個方向（4 旋轉 × 鏡像/非鏡像），去重
    fn compute_all_variants(base: &PieceVariant) -> Vec<PieceVariant> {
        let mut variants = Vec::with_capacity(8);
        let mut current = base.clone();
        for _ in 0..4 {
            variants.push(current.clone());
            variants.push(current.mirrored());
            current = current.rotated();
        }
        variants.sort();
        variants.dedup();
        variants
    }
}

/// 將細胞集合標準化至左上角 (0,0)
fn normalize(cells: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    let min_x = cells.iter().map(|&(x, _)| x).min().unwrap_or(0);
    let min_y = cells.iter().map(|&(_, y)| y).min().unwrap_or(0);
    let mut normalized: Vec<(i32, i32)> = cells
        .iter()
        .map(|&(x, y)| (x - min_x, y - min_y))
        .collect();
    normalized.sort();
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let cells = vec![(3, 1), (2, 0), (4, 2)];
        let norm = normalize(cells);
        assert_eq!(norm, vec![(0, 0), (1, 1), (2, 2)]);
    }

    #[test]
    fn test_rotation_90() {
        let v = PieceVariant::new(vec![(0, 0), (1, 0), (0, 1)]);
        let rotated = v.rotated();
        // L triomino rotated 90° CW: cells are (0,0), (0,1), (1,1)
        assert_eq!(rotated.cells.len(), 3);
        assert!(rotated.cells.contains(&(0, 0)));
        assert!(rotated.cells.contains(&(0, 1)));
        assert!(rotated.cells.contains(&(1, 1)));
    }

    #[test]
    fn test_mirror() {
        let v = PieceVariant::new(vec![(0, 0), (1, 0), (0, 1)]);
        let mirrored = v.mirrored();
        // L triomino mirrored: cells are (0,0), (1,0), (1,1)
        assert_eq!(mirrored.cells.len(), 3);
        assert!(mirrored.cells.contains(&(0, 0)));
        assert!(mirrored.cells.contains(&(1, 0)));
        assert!(mirrored.cells.contains(&(1, 1)));
    }

    #[test]
    fn test_all_rotations_of_line() {
        // I 形 (直線 3): (0,0),(0,1),(0,2)
        let v = PieceVariant::new(vec![(0, 0), (0, 1), (0, 2)]);
        let rots = v.all_rotations();
        // 直線 3 有 2 種不重複旋轉：垂直與水平
        assert_eq!(rots.len(), 2);
    }

    #[test]
    fn test_all_rotations_of_square() {
        // 方形: (0,0),(1,0),(0,1),(1,1)
        let v = PieceVariant::new(vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
        let rots = v.all_rotations();
        // 方形所有旋轉都相同 → 1 種
        assert_eq!(rots.len(), 1);
    }

    #[test]
    fn test_piece_shape_variants() {
        let shape = PieceShape::new(PieceId(0), "Monomino", vec![(0, 0)]);
        assert!(!shape.variants.is_empty());
        // monomino 所有方向都相同
        assert_eq!(shape.variants.len(), 1);
    }
}
