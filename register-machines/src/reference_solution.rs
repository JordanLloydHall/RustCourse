use std::collections::HashMap;

pub type Label = usize;
pub type Register = u128;
pub type State = (Label, HashMap<Register, u128>);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Instruction {
    Add(Register, Label),
    Sub(Register, Label, Label),
    Halt,
}

use Instruction::*;
pub fn eval_program(program: &[Instruction], state: &State) -> State {
    let (mut instr, mut registers) = state.clone();

    while instr < program.len() {
        match program[instr] {
            Add(r, l) => {
                registers.entry(r).and_modify(|v| *v += 1).or_insert(1);
                instr = l;
            }
            Sub(r, l1, l2) => {
                let v = registers.entry(r).or_insert(0);
                if *v != 0 {
                    *v -= 1;
                    instr = l1;
                } else {
                    instr = l2;
                }
            }
            Halt => break,
        }
    }

    (instr, registers)
}

fn encode_pair1(x: u128, y: u128) -> u128 {
    2u128.pow(x as u32) * (2 * y + 1)
}

fn encode_pair2(x: u128, y: u128) -> u128 {
    encode_pair1(x, y) - 1
}

fn encode_program_to_list(program: &[Instruction]) -> Vec<u128> {
    program
        .iter()
        .map(|p| match p {
            Add(r, l) => encode_pair1(2 * *r, *l as u128),
            Sub(r, l1, l2) => encode_pair1(2 * *r + 1, encode_pair2(*l1 as u128, *l2 as u128)),
            Halt => 0,
        })
        .collect()
}

pub fn decode_pair1(a: u128) -> (u128, u128) {
    let x = a.trailing_zeros() as u128; 
    let z = a >> x;
    let y = (z - 1) / 2;
    (x, y)
}

fn decode_pair2(a: u128) -> (u128, u128) {
    decode_pair1(a + 1)
}

pub fn decode_list_to_program(program: &[u128]) -> Vec<Instruction> {
    program
        .iter()
        .map(|p| {
            if *p == 0 {
                Halt
            } else {
                let (y, z) = decode_pair1(*p);
                if (y % 2) == 0 {
                    Add(y / 2, z as usize)
                } else {
                    let (j, k) = decode_pair2(z);
                    Sub((y - 1) / 2, j as usize, k as usize)
                }
            }
        })
        .collect()
}

pub fn decode_godel_to_list(g: u128) -> Vec<u128> {
    let mut ret_vec = Vec::new();
    let mut tmp = g;

    while tmp != 0 {
        let (x, tmp2) = decode_pair1(tmp);
        tmp = tmp2;
        ret_vec.push(x);
    }

    ret_vec
}

pub fn encode_list_to_godel(l: &[u128]) -> u128 {
    l.iter().rev().fold(0, |acc, v| encode_pair1(*v, acc))
}

fn main() {

}

#[test]
fn godel_num_to_godel_list() {
    let n = 2u128.pow(46) * 20483;
    let godel_list = decode_godel_to_list(n);
    let true_godel_list = vec![46, 0, 10, 1];
    assert_eq!(godel_list, true_godel_list)
}

#[test]
fn godel_list_to_godel_num() {
    let godel_num = encode_list_to_godel(&[46, 0, 10, 1]);
    assert_eq!(godel_num, 2u128.pow(46) * 20483)
}

#[test]
fn godel_list_to_program() {
    let program = decode_list_to_program(&vec![46, 0, 10, 1]);
    assert_eq!(program, vec![Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)])
}

#[test]
fn program_to_godel_list() {
    let program = encode_program_to_list(&[Sub(0, 2, 1), Halt, Sub(0, 0, 1), Add(0, 0)]);
    assert_eq!(program, [46, 0, 10, 1,])
}

#[test]
fn program_produces_correct_state() {
    use std::array::IntoIter;
    let program = vec![
        Sub(1, 2, 1),
        Halt,
        Sub(1, 3, 4),
        Sub(1, 5, 4),
        Halt,
        Add(0, 0),
    ];
    let final_state = eval_program(
        &program,
        &(
            0,
            HashMap::<_, _>::from_iter(IntoIter::new([(1, 7)])),
        ),
    );
    assert_eq!(
        final_state,
        (
            4,
            HashMap::<_, _>::from_iter(IntoIter::new([(0, 2), (1, 0)]))
        )
    )
}
