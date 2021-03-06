extern crate ndarray;
extern crate itertools;

use ndarray::Array;
use ndarray::{Ix, Si, S};
use ndarray::{
    ArrayBase,
    Data,
    Dimension,
    aview1,
};

use itertools::assert_equal;

#[test]
fn double_ended()
{
    let a = Array::linspace(0., 7., 8);
    let mut it = a.iter().map(|x| *x);
    assert_eq!(it.next(), Some(0.));
    assert_eq!(it.next_back(), Some(7.));
    assert_eq!(it.next(), Some(1.));
    assert_eq!(it.rev().last(), Some(2.));
}

#[test]
fn indexed()
{
    let a = Array::linspace(0., 7., 8);
    for (i, elt) in a.indexed_iter() {
        assert_eq!(i, *elt as Ix);
    }
    let a = a.reshape((2, 4, 1));
    let (mut i, mut j, k) = (0, 0, 0);
    for (idx, elt) in a.indexed_iter() {
        assert_eq!(idx, (i, j, k));
        j += 1;
        if j == 4 {
            j = 0;
            i += 1;
        }
        println!("{:?}", (idx, elt));
    }
}


fn assert_slice_correct<A, S, D>(v: &ArrayBase<S, D>)
    where S: Data<Elem=A>,
          D: Dimension,
          A: PartialEq + std::fmt::Debug,
{
    let slc = v.as_slice();
    assert!(slc.is_some());
    let slc = slc.unwrap();
    assert_eq!(v.len(), slc.len());
    assert_equal(v.iter(), slc);
}

#[test]
fn as_slice() {
    let a = Array::linspace(0., 7., 8);
    let a = a.reshape((2, 4, 1));

    assert_slice_correct(&a);

    let a = a.reshape((2, 4));
    assert_slice_correct(&a);

    assert!(a.view().subview(1, 0).as_slice().is_none());

    let v = a.view();
    assert_slice_correct(&v);
    assert_slice_correct(&v.subview(0, 0));
    assert_slice_correct(&v.subview(0, 1));

    assert!(v.slice(&[S, Si(0, Some(1), 1)]).as_slice().is_none());
    println!("{:?}", v.slice(&[Si(0, Some(1), 2), S]));
    assert!(v.slice(&[Si(0, Some(1), 2), S]).as_slice().is_some());

    // `u` is contiguous, because the column stride of `2` doesn't matter
    // when the result is just one row anyway -- length of that dimension is 1
    let u = v.slice(&[Si(0, Some(1), 2), S]);
    println!("{:?}", u.shape());
    println!("{:?}", u.strides());
    println!("{:?}", v.slice(&[Si(0, Some(1), 2), S]));
    assert!(u.as_slice().is_some());
    assert_slice_correct(&u);

    let a = a.reshape((8, 1));
    assert_slice_correct(&a);
    let u = a.slice(&[Si(0, None, 2), S]);
    println!("u={:?}, shape={:?}, strides={:?}", u, u.shape(), u.strides());
    assert!(u.as_slice().is_none());
}

#[test]
fn inner_iter() {
    let a = Array::from_iter(0..12);
    let a = a.reshape((2, 3, 2));
    // [[[0, 1],
    //   [2, 3],
    //   [4, 5]],
    //  [[6, 7],
    //   [8, 9],
    //    ...
    assert_equal(a.inner_iter(),
                 vec![aview1(&[0, 1]), aview1(&[2, 3]), aview1(&[4, 5]),
                      aview1(&[6, 7]), aview1(&[8, 9]), aview1(&[10, 11])]);
    let mut b = Array::zeros((2, 3, 2));
    b.swap_axes(0, 2);
    b.assign(&a);
    assert_equal(b.inner_iter(),
                 vec![aview1(&[0, 1]), aview1(&[2, 3]), aview1(&[4, 5]),
                      aview1(&[6, 7]), aview1(&[8, 9]), aview1(&[10, 11])]);
}

#[test]
fn inner_iter_corner_cases() {
    let a0 = Array::zeros(());
    assert_equal(a0.inner_iter(), vec![aview1(&[0])]);

    let a2 = Array::<i32, _>::zeros((0, 3));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 0]);

    let a2 = Array::<i32, _>::zeros((3, 0));
    assert_equal(a2.inner_iter(),
                 vec![aview1(&[]); 3]);
}
