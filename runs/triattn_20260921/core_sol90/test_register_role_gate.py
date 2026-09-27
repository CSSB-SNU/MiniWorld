"""Regression against actual SASS of the observed register-allocation deadlock."""
from pathlib import Path
import unittest
from register_role_gate import check_register_roles

HERE=Path(__file__).resolve().parent


class ActualRegisterAllocation(unittest.TestCase):
    def read(self,name):
        return (HERE/(name+'_screen/ru5.sass')).read_text()

    def test_installed_control_fits(self):
        self.assertEqual(check_register_roles(self.read('ptxas_current'))['role_register_pool'],65536)

    def test_real_mixed_24_fits(self):
        result=check_register_roles(self.read('leanprod24q168'),dict(producer=24,consumers=[168,160,160]))
        self.assertEqual(result['role_register_pool'],65536)

    def test_deadlock_binary_rejects_intended_roles(self):
        with self.assertRaisesRegex(ValueError,'Emitted register roles differ'):
            check_register_roles(self.read('leanprod24q168b8'),dict(producer=24,consumers=[168,160,160]))

    def test_deadlock_binary_rejects_actual_pool(self):
        with self.assertRaisesRegex(ValueError,'Register pool overflow'):
            check_register_roles(self.read('leanprod24q168b8'),dict(producer=32,consumers=[168,160,160]))


if __name__=='__main__':
    unittest.main()
